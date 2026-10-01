"""Offline checks for the range-based Zenodo subset fetcher."""

from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import fetch_zenodo_erp_subset as fetch


def _write_zip(path: Path, members: dict[str, bytes], *, stored: bool = False) -> None:
    compression = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


class _FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        headers: dict[str, str],
        url: str,
    ) -> None:
        self._payload = payload
        self.status = status
        self.headers = headers
        self._url = url
        self.closed = False

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            result = self._payload
            self._payload = b""
            return result
        result = self._payload[:amount]
        self._payload = self._payload[amount:]
        return result

    def close(self) -> None:
        self.closed = True


class ZenodoSubsetFetcherTests(unittest.TestCase):
    def test_list_and_dry_run_do_not_write_or_extract_members(self) -> None:
        members = {name: name.encode() for name in fetch.SELECTED_MEMBERS}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.zip"
            output = root / "outputs"
            _write_zip(archive, members)
            metadata = fetch.ArchiveMetadata("https://zenodo.org/fixture", archive.stat().st_size, fetch.ARCHIVE_MD5)
            for option in ("--list", "--dry-run"):
                stream = io.StringIO()
                with patch.object(fetch, "fetch_record_metadata", return_value=metadata), \
                     patch.object(fetch, "HTTPRangeReader", return_value=fetch.LocalRangeReader(archive)), \
                     patch.object(fetch, "execute_plans") as execute, redirect_stdout(stream):
                    self.assertEqual(fetch.main([option, "--output-root", str(output)]), 0)
                execute.assert_not_called()
                self.assertFalse(output.exists())
                self.assertIn("P7_with_erp_lat_bi.vtk", stream.getvalue())

    def test_declared_subset_is_exact_and_excludes_right_atrium(self) -> None:
        self.assertEqual(len(fetch.SELECTED_MEMBERS), 19)
        self.assertEqual(len(set(fetch.SELECTED_MEMBERS)), 19)
        self.assertFalse(any("/P2/" in name for name in fetch.SELECTED_MEMBERS))
        for patient in fetch.LA_PATIENTS:
            expected = [
                name
                for name in fetch.SELECTED_MEMBERS
                if f"/{patient}/" in name
            ]
            self.assertEqual(len(expected), 3)

    def test_pinned_record_metadata(self) -> None:
        payload = {
            "id": fetch.RECORD_ID,
            "doi": fetch.RECORD_DOI,
            "files": [
                {
                    "key": "data.zip",
                    "size": fetch.ARCHIVE_SIZE,
                    "checksum": f"md5:{fetch.ARCHIVE_MD5}",
                    "links": {
                        "self": (
                            "https://zenodo.org/api/records/10726677/"
                            "files/data.zip/content"
                        )
                    },
                }
            ],
        }
        metadata = fetch.validate_record_payload(payload)
        self.assertEqual(metadata.size, fetch.ARCHIVE_SIZE)
        self.assertEqual(metadata.md5, fetch.ARCHIVE_MD5)

        payload["files"][0]["checksum"] = "md5:" + "0" * 32
        with self.assertRaisesRegex(fetch.FetchError, "checksum disagreement"):
            fetch.validate_record_payload(payload)

    def test_small_zip_parse_extract_skip_and_explicit_replace(self) -> None:
        members = {
            "data/meshes/element_tag.csv": b"tag,name\n1,atrium\n",
            "data/meshes/P1/P1_with_erp_lat_bi.vtk": b"vtk payload\n" * 100,
        }
        selected_names = tuple(members)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive_path = base / "small.zip"
            output_root = base / "output"
            _write_zip(archive_path, members)
            reader = fetch.LocalRangeReader(archive_path)
            entries = fetch.read_central_directory(reader)
            selected = fetch.select_declared_entries(entries, selected_names)

            plans = fetch.plan_destinations(
                selected, output_root, overwrite_mismatched=False
            )
            self.assertTrue(all(plan.status == "download" for plan in plans))
            fetch.execute_plans(reader, plans)
            for name, payload in members.items():
                relative = name.removeprefix(fetch.ARCHIVE_PREFIX)
                self.assertEqual((output_root / relative).read_bytes(), payload)

            verified = fetch.plan_destinations(
                selected, output_root, overwrite_mismatched=False
            )
            self.assertTrue(all(plan.status == "verified" for plan in verified))

            target = output_root / "element_tag.csv"
            target.write_bytes(b"not the archive member")
            with self.assertRaisesRegex(fetch.FetchError, "--overwrite-mismatched"):
                fetch.plan_destinations(
                    selected, output_root, overwrite_mismatched=False
                )
            replacements = fetch.plan_destinations(
                selected, output_root, overwrite_mismatched=True
            )
            self.assertEqual(replacements[0].status, "replace")
            fetch.execute_plans(reader, replacements)
            self.assertEqual(target.read_bytes(), members["data/meshes/element_tag.csv"])

    def test_member_crc_is_checked_after_decompression(self) -> None:
        name = "data/meshes/element_tag.csv"
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "stored.zip"
            _write_zip(archive_path, {name: b"abcdef"}, stored=True)
            initial_reader = fetch.LocalRangeReader(archive_path)
            entry = fetch.read_central_directory(initial_reader)[name]

            # Corrupt one stored data byte while leaving both ZIP headers intact.
            with archive_path.open("r+b") as stream:
                stream.seek(entry.local_header_offset)
                header = stream.read(fetch.LOCAL_FIXED_SIZE)
                fields = fetch.struct.unpack("<4s5H3I2H", header)
                data_offset = (
                    entry.local_header_offset
                    + fetch.LOCAL_FIXED_SIZE
                    + fields[9]
                    + fields[10]
                )
                stream.seek(data_offset + 2)
                original = stream.read(1)
                stream.seek(data_offset + 2)
                stream.write(bytes((original[0] ^ 0x20,)))

            corrupted_reader = fetch.LocalRangeReader(archive_path)
            with self.assertRaisesRegex(fetch.FetchError, "CRC-32 disagreement"):
                fetch.read_member_payload(corrupted_reader, entry)

    def test_wrong_central_crc_or_unsafe_path_fails_closed(self) -> None:
        safe_name = "data/meshes/element_tag.csv"
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            safe_archive = base / "safe.zip"
            _write_zip(safe_archive, {safe_name: b"1234"})
            reader = fetch.LocalRangeReader(safe_archive)
            entry = fetch.read_central_directory(reader)[safe_name]
            with self.assertRaisesRegex(fetch.FetchError, "local/central"):
                fetch.read_member_payload(reader, replace(entry, crc32=0))

            unsafe_archive = base / "unsafe.zip"
            _write_zip(unsafe_archive, {"../escape.txt": b"forbidden"})
            with self.assertRaisesRegex(fetch.FetchError, "unsafe ZIP member"):
                fetch.read_central_directory(fetch.LocalRangeReader(unsafe_archive))

    def test_http_reader_requires_exact_206_content_range(self) -> None:
        url = "https://zenodo.org/api/records/10726677/files/data.zip/content"
        reader = fetch.HTTPRangeReader(url, 10, retries=0)
        ignored = _FakeResponse(
            b"0123456789",
            status=200,
            headers={"Content-Length": "10"},
            url=url,
        )
        with patch.object(fetch, "_open_with_retries", return_value=ignored):
            with self.assertRaisesRegex(fetch.FetchError, "ignored byte range"):
                reader.read_range(2, 4)
        self.assertTrue(ignored.closed)

        exact = _FakeResponse(
            b"234",
            status=206,
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 2-4/10",
            },
            url=url,
        )
        with patch.object(fetch, "_open_with_retries", return_value=exact):
            self.assertEqual(reader.read_range(2, 4), b"234")
        self.assertTrue(exact.closed)


if __name__ == "__main__":
    unittest.main()
