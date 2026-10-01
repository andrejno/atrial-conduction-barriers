"""Fetch the small, declared Zenodo subset used by the surface experiment.

The public archive for Zenodo record 10726677 is about 1.6 GB.  This utility
reads its ZIP end record and central directory with HTTP byte ranges, then
downloads only the nineteen declared members needed by the patient-surface
experiment.  It deliberately does not rely on hand-copied byte offsets.

Integrity is checked at three levels:

* the official Zenodo API must report the pinned ``data.zip`` byte length and
  MD5 digest;
* every HTTP response must carry the exact requested ``Content-Range``; and
* every extracted member must match its central-directory length and CRC-32.

The API MD5 check pins the identity of the remote archive without downloading
all 1.6 GB.  ``--verify-full-archive-md5`` is available when a byte-for-byte
MD5 pass over the complete remote archive is required; it is intentionally not
the default because it transfers the whole archive.

Output paths are rooted directly below ``--output-root``.  For example,
``data/meshes/P1/P1_with_erp_lat_bi.vtk`` becomes
``OUTPUT_ROOT/P1/P1_with_erp_lat_bi.vtk``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zlib


RECORD_ID = 10_726_677
RECORD_DOI = "10.5281/zenodo.10726677"
RECORD_API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
ARCHIVE_KEY = "data.zip"
ARCHIVE_SIZE = 1_624_523_251
ARCHIVE_MD5 = "97d6ae8a62c2a5fdc3a61526942cb23d"
ARCHIVE_PREFIX = "data/meshes/"
DEFAULT_OUTPUT_ROOT = Path("external_data/zenodo_erp/meshes")
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 3
USER_AGENT = "cardiac-sign-graph-reproducibility/1.0"

LA_PATIENTS = ("P1", "P3", "P4", "P5", "P6", "P7")
SELECTED_MEMBERS = (
    "data/meshes/element_tag.csv",
    *(
        member
        for patient in LA_PATIENTS
        for member in (
            f"data/meshes/{patient}/{patient}_with_erp_lat_bi.vtk",
            f"data/meshes/{patient}/bilayer/LA_bilayer_with_fiber_um.pts",
            f"data/meshes/{patient}/bilayer/LA_bilayer_with_fiber_um.elem",
        )
    ),
)

EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"
EOCD_FIXED_SIZE = 22
EOCD_MAX_SEARCH = EOCD_FIXED_SIZE + 65_535
CENTRAL_FIXED_SIZE = 46
LOCAL_FIXED_SIZE = 30
ZIP64_U16 = 0xFFFF
ZIP64_U32 = 0xFFFFFFFF
RETRYABLE_HTTP_CODES = frozenset((429, 500, 502, 503, 504))


class FetchError(RuntimeError):
    """Raised when remote metadata, ZIP structure, or extracted data disagree."""


@dataclass(frozen=True)
class ArchiveMetadata:
    url: str
    size: int
    md5: str


@dataclass(frozen=True)
class EndOfCentralDirectory:
    entries: int
    central_size: int
    central_offset: int


@dataclass(frozen=True)
class ZipEntry:
    name: str
    flags: int
    compression: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    external_attributes: int


@dataclass(frozen=True)
class DestinationPlan:
    entry: ZipEntry
    destination: Path
    status: str


class RangeReader(Protocol):
    """Minimal random-access source used by the ZIP parser."""

    size: int

    def read_range(self, start: int, end: int) -> bytes:
        """Return the inclusive byte range ``start`` through ``end``."""


class LocalRangeReader:
    """Exact range reader for tests and already-downloaded archives."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.size = path.stat().st_size
        except OSError as exc:
            raise FetchError(f"cannot stat local archive {path}: {exc}") from exc

    def read_range(self, start: int, end: int) -> bytes:
        expected = _validate_range(start, end, self.size)
        try:
            with self.path.open("rb") as stream:
                stream.seek(start)
                payload = stream.read(expected)
        except OSError as exc:
            raise FetchError(
                f"cannot read bytes {start}-{end} from {self.path}: {exc}"
            ) from exc
        if len(payload) != expected:
            raise FetchError(
                f"short local read for bytes {start}-{end}: "
                f"expected {expected}, received {len(payload)}"
            )
        return payload


class HTTPRangeReader:
    """HTTP reader that accepts only exact 206 byte-range responses."""

    def __init__(
        self,
        url: str,
        size: int,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        _validate_zenodo_url(url, "archive content URL")
        if size <= 0:
            raise FetchError(f"invalid remote archive size: {size}")
        if timeout <= 0:
            raise FetchError("timeout must be positive")
        if retries < 0:
            raise FetchError("retries cannot be negative")
        self.url = url
        self.size = size
        self.timeout = timeout
        self.retries = retries

    def read_range(self, start: int, end: int) -> bytes:
        expected = _validate_range(start, end, self.size)
        request = Request(
            self.url,
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
                "User-Agent": USER_AGENT,
            },
        )
        response = _open_with_retries(
            request, timeout=self.timeout, retries=self.retries
        )
        try:
            status_code = getattr(response, "status", response.getcode())
            final_url = response.geturl()
            _validate_zenodo_url(final_url, "archive response URL")
            if status_code != 206:
                raise FetchError(
                    f"server ignored byte range {start}-{end}: HTTP {status_code}"
                )
            encoding = response.headers.get("Content-Encoding", "identity").lower()
            if encoding not in ("", "identity"):
                raise FetchError(f"unexpected Content-Encoding for range: {encoding}")
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if match is None:
                raise FetchError(
                    f"missing or malformed Content-Range: {content_range!r}"
                )
            actual_start, actual_end, total = map(int, match.groups())
            if (actual_start, actual_end, total) != (start, end, self.size):
                raise FetchError(
                    "Content-Range disagreement: "
                    f"expected bytes {start}-{end}/{self.size}, "
                    f"received {content_range!r}"
                )
            declared_length = response.headers.get("Content-Length")
            if declared_length is None or not declared_length.isdigit():
                raise FetchError("range response lacks a numeric Content-Length")
            if int(declared_length) != expected:
                raise FetchError(
                    f"Content-Length disagreement for bytes {start}-{end}: "
                    f"expected {expected}, received {declared_length}"
                )
            payload = response.read(expected + 1)
        finally:
            response.close()
        if len(payload) != expected:
            raise FetchError(
                f"range length disagreement for bytes {start}-{end}: "
                f"expected {expected}, received {len(payload)}"
            )
        return payload


def _validate_range(start: int, end: int, size: int) -> int:
    if start < 0 or end < start or end >= size:
        raise FetchError(f"invalid byte range {start}-{end} for size {size}")
    return end - start + 1


def _validate_zenodo_url(url: str, field: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "zenodo.org" or host.endswith(".zenodo.org")
    ):
        raise FetchError(f"{field} is not an official HTTPS Zenodo URL: {url!r}")


def _retry_delay(error: HTTPError, attempt: int) -> float:
    header = error.headers.get("Retry-After") if error.headers is not None else None
    if header and header.isdigit():
        return min(float(header), 60.0)
    return min(2.0**attempt, 8.0)


def _open_with_retries(request: Request, *, timeout: float, retries: int) -> Any:
    for attempt in range(retries + 1):
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_CODES or attempt == retries:
                raise FetchError(
                    f"HTTP request failed for {request.full_url}: {exc}"
                ) from exc
            time.sleep(_retry_delay(exc, attempt))
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise FetchError(
                    f"network request failed for {request.full_url}: {exc}"
                ) from exc
            time.sleep(min(2.0**attempt, 8.0))
    raise AssertionError("retry loop terminated unexpectedly")


def fetch_record_metadata(
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> ArchiveMetadata:
    """Read and strictly validate the pinned record's ``data.zip`` metadata."""
    request = Request(
        RECORD_API_URL,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    response = _open_with_retries(request, timeout=timeout, retries=retries)
    try:
        status_code = getattr(response, "status", response.getcode())
        _validate_zenodo_url(response.geturl(), "record response URL")
        if status_code != 200:
            raise FetchError(f"Zenodo record API returned HTTP {status_code}")
        raw = response.read(2_000_001)
    finally:
        response.close()
    if len(raw) > 2_000_000:
        raise FetchError("Zenodo record metadata unexpectedly exceeds 2 MB")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FetchError(f"Zenodo record API did not return valid JSON: {exc}") from exc
    return validate_record_payload(payload)


def validate_record_payload(payload: Any) -> ArchiveMetadata:
    """Validate API JSON separately so the checks can be unit-tested offline."""
    if not isinstance(payload, Mapping):
        raise FetchError("Zenodo record response must be a JSON object")
    if payload.get("id") != RECORD_ID:
        raise FetchError(
            f"Zenodo record id disagreement: expected {RECORD_ID}, "
            f"received {payload.get('id')!r}"
        )
    if payload.get("doi") != RECORD_DOI:
        raise FetchError(
            f"Zenodo DOI disagreement: expected {RECORD_DOI!r}, "
            f"received {payload.get('doi')!r}"
        )
    files = payload.get("files")
    if not isinstance(files, list):
        raise FetchError("Zenodo record response has no files list")
    matches = [item for item in files if isinstance(item, Mapping) and item.get("key") == ARCHIVE_KEY]
    if len(matches) != 1:
        raise FetchError(
            f"expected exactly one {ARCHIVE_KEY!r} entry, found {len(matches)}"
        )
    item = matches[0]
    size = item.get("size")
    checksum = item.get("checksum")
    links = item.get("links")
    if size != ARCHIVE_SIZE:
        raise FetchError(
            f"{ARCHIVE_KEY} size disagreement: expected {ARCHIVE_SIZE}, "
            f"received {size!r}"
        )
    expected_checksum = f"md5:{ARCHIVE_MD5}"
    if checksum != expected_checksum:
        raise FetchError(
            f"{ARCHIVE_KEY} checksum disagreement: expected {expected_checksum!r}, "
            f"received {checksum!r}"
        )
    if not isinstance(links, Mapping) or not isinstance(links.get("self"), str):
        raise FetchError(f"{ARCHIVE_KEY} has no content URL")
    url = links["self"]
    _validate_zenodo_url(url, "archive content URL")
    return ArchiveMetadata(url=url, size=size, md5=ARCHIVE_MD5)


def read_end_of_central_directory(reader: RangeReader) -> EndOfCentralDirectory:
    """Locate and validate the classic ZIP end-of-central-directory record."""
    if reader.size < EOCD_FIXED_SIZE:
        raise FetchError("archive is too short to contain a ZIP end record")
    tail_size = min(reader.size, EOCD_MAX_SEARCH)
    tail_start = reader.size - tail_size
    tail = reader.read_range(tail_start, reader.size - 1)
    search_end = len(tail)
    while True:
        position = tail.rfind(EOCD_SIGNATURE, 0, search_end)
        if position < 0:
            raise FetchError("ZIP end-of-central-directory record was not found")
        if position + EOCD_FIXED_SIZE <= len(tail):
            values = struct.unpack_from("<4s4H2IH", tail, position)
            (
                _signature,
                disk_number,
                central_disk,
                entries_on_disk,
                total_entries,
                central_size,
                central_offset,
                comment_length,
            ) = values
            if position + EOCD_FIXED_SIZE + comment_length == len(tail):
                break
        search_end = position
    if disk_number != 0 or central_disk != 0 or entries_on_disk != total_entries:
        raise FetchError("multi-disk ZIP archives are not supported")
    if (
        total_entries == ZIP64_U16
        or central_size == ZIP64_U32
        or central_offset == ZIP64_U32
    ):
        raise FetchError("ZIP64 central directories are not supported by this fetcher")
    eocd_offset = tail_start + position
    if central_offset + central_size != eocd_offset:
        raise FetchError(
            "central-directory extent does not end at the ZIP end record: "
            f"offset={central_offset}, size={central_size}, eocd={eocd_offset}"
        )
    if total_entries <= 0 or central_size <= 0:
        raise FetchError("ZIP central directory is empty")
    return EndOfCentralDirectory(
        entries=total_entries,
        central_size=central_size,
        central_offset=central_offset,
    )


def _decode_member_name(raw: bytes, flags: int) -> str:
    encoding = "utf-8" if flags & (1 << 11) else "cp437"
    try:
        name = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise FetchError(f"cannot decode ZIP member name as {encoding}: {exc}") from exc
    _validate_member_name(name)
    return name


def _validate_member_name(name: str) -> None:
    if not name or "\x00" in name or "\\" in name:
        raise FetchError(f"unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise FetchError(f"unsafe ZIP member path: {name!r}")


def read_central_directory(reader: RangeReader) -> dict[str, ZipEntry]:
    """Read all central records and return an exact name-to-entry mapping."""
    eocd = read_end_of_central_directory(reader)
    central = reader.read_range(
        eocd.central_offset, eocd.central_offset + eocd.central_size - 1
    )
    entries: dict[str, ZipEntry] = {}
    position = 0
    for index in range(eocd.entries):
        if position + CENTRAL_FIXED_SIZE > len(central):
            raise FetchError(f"central record {index} is truncated")
        values = struct.unpack_from("<4s6H3I5H2I", central, position)
        if values[0] != CENTRAL_SIGNATURE:
            raise FetchError(
                f"invalid central signature at entry {index}, byte {position}"
            )
        flags = values[3]
        compression = values[4]
        crc32 = values[7]
        compressed_size = values[8]
        uncompressed_size = values[9]
        name_length = values[10]
        extra_length = values[11]
        comment_length = values[12]
        disk_start = values[13]
        external_attributes = values[15]
        local_header_offset = values[16]
        record_end = (
            position
            + CENTRAL_FIXED_SIZE
            + name_length
            + extra_length
            + comment_length
        )
        if record_end > len(central):
            raise FetchError(f"central record {index} extends beyond the directory")
        raw_name = central[
            position + CENTRAL_FIXED_SIZE : position + CENTRAL_FIXED_SIZE + name_length
        ]
        name = _decode_member_name(raw_name, flags)
        if name in entries:
            raise FetchError(f"duplicate ZIP member name: {name!r}")
        if disk_start != 0:
            raise FetchError(f"member {name!r} starts on a different ZIP disk")
        if (
            compressed_size == ZIP64_U32
            or uncompressed_size == ZIP64_U32
            or local_header_offset == ZIP64_U32
        ):
            raise FetchError(f"ZIP64 member is unsupported: {name!r}")
        entries[name] = ZipEntry(
            name=name,
            flags=flags,
            compression=compression,
            crc32=crc32,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_header_offset=local_header_offset,
            external_attributes=external_attributes,
        )
        position = record_end
    if position != len(central):
        raise FetchError(
            f"unparsed bytes remain in central directory: {len(central) - position}"
        )
    return entries


def select_declared_entries(
    entries: Mapping[str, ZipEntry],
    selected_names: Sequence[str] = SELECTED_MEMBERS,
) -> list[ZipEntry]:
    if len(selected_names) != len(set(selected_names)):
        raise FetchError("declared member list contains duplicates")
    missing = [name for name in selected_names if name not in entries]
    if missing:
        raise FetchError("declared archive member(s) missing: " + ", ".join(missing))
    selected = [entries[name] for name in selected_names]
    for entry in selected:
        if entry.flags & 1:
            raise FetchError(f"encrypted ZIP member is unsupported: {entry.name!r}")
        if entry.compression not in (0, 8):
            raise FetchError(
                f"unsupported compression method {entry.compression} "
                f"for {entry.name!r}"
            )
        unix_mode = (entry.external_attributes >> 16) & 0xFFFF
        if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
            raise FetchError(f"symbolic-link ZIP member is forbidden: {entry.name!r}")
        if entry.name.endswith("/"):
            raise FetchError(f"declared member is a directory: {entry.name!r}")
    return selected


def _entry_destination(output_root: Path, entry: ZipEntry) -> Path:
    if not entry.name.startswith(ARCHIVE_PREFIX):
        raise FetchError(
            f"declared member does not start with {ARCHIVE_PREFIX!r}: {entry.name!r}"
        )
    relative_text = entry.name[len(ARCHIVE_PREFIX) :]
    _validate_member_name(relative_text)
    relative = PurePosixPath(relative_text)
    return output_root.joinpath(*relative.parts)


def _file_crc32(path: Path) -> tuple[int, int]:
    checksum = 0
    size = 0
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                checksum = zlib.crc32(block, checksum)
    except OSError as exc:
        raise FetchError(f"cannot check existing file {path}: {exc}") from exc
    return size, checksum & 0xFFFFFFFF


def plan_destinations(
    selected: Iterable[ZipEntry],
    output_root: Path,
    *,
    overwrite_mismatched: bool,
    fail_on_conflict: bool = True,
) -> list[DestinationPlan]:
    """Inspect every target before any network payload is downloaded or written."""
    root = output_root.expanduser().resolve(strict=False)
    plans: list[DestinationPlan] = []
    conflicts: list[str] = []
    for entry in selected:
        destination = _entry_destination(root, entry)
        resolved = destination.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise FetchError(f"target escapes output root: {destination}")
        if destination.is_symlink():
            raise FetchError(f"refusing symbolic-link destination: {destination}")
        if not destination.exists():
            status = "download"
        elif not destination.is_file():
            raise FetchError(f"target exists but is not a regular file: {destination}")
        else:
            size, checksum = _file_crc32(destination)
            if (size, checksum) == (entry.uncompressed_size, entry.crc32):
                status = "verified"
            elif overwrite_mismatched:
                status = "replace"
            else:
                status = "conflict"
                conflicts.append(
                    f"{destination} (found size={size}, crc32={checksum:08x}; "
                    f"expected size={entry.uncompressed_size}, crc32={entry.crc32:08x})"
                )
        plans.append(DestinationPlan(entry, destination, status))
    if conflicts and fail_on_conflict:
        detail = "\n  ".join(conflicts)
        raise FetchError(
            "mismatched output file(s) would be overwritten; rerun with "
            f"--overwrite-mismatched only after reviewing them:\n  {detail}"
        )
    return plans


def read_member_payload(reader: RangeReader, entry: ZipEntry) -> bytes:
    """Fetch, decompress, and CRC-check one member described by the central index."""
    header_offset = entry.local_header_offset
    header = reader.read_range(header_offset, header_offset + LOCAL_FIXED_SIZE - 1)
    values = struct.unpack("<4s5H3I2H", header)
    if values[0] != LOCAL_SIGNATURE:
        raise FetchError(f"invalid local header signature for {entry.name!r}")
    local_flags = values[2]
    local_compression = values[3]
    local_crc32 = values[6]
    local_compressed_size = values[7]
    local_uncompressed_size = values[8]
    name_length = values[9]
    extra_length = values[10]
    if local_flags != entry.flags:
        raise FetchError(f"local/central flag disagreement for {entry.name!r}")
    if local_compression != entry.compression:
        raise FetchError(f"local/central compression disagreement for {entry.name!r}")
    variable_size = name_length + extra_length
    if variable_size:
        variable = reader.read_range(
            header_offset + LOCAL_FIXED_SIZE,
            header_offset + LOCAL_FIXED_SIZE + variable_size - 1,
        )
    else:
        variable = b""
    local_name = _decode_member_name(variable[:name_length], local_flags)
    if local_name != entry.name:
        raise FetchError(
            f"local/central name disagreement: {local_name!r} != {entry.name!r}"
        )
    uses_descriptor = bool(entry.flags & (1 << 3))
    if not uses_descriptor and (
        local_crc32 != entry.crc32
        or local_compressed_size != entry.compressed_size
        or local_uncompressed_size != entry.uncompressed_size
    ):
        raise FetchError(f"local/central size or CRC disagreement for {entry.name!r}")
    data_offset = header_offset + LOCAL_FIXED_SIZE + variable_size
    if entry.compressed_size:
        compressed = reader.read_range(
            data_offset, data_offset + entry.compressed_size - 1
        )
    else:
        compressed = b""
    if entry.compression == 0:
        payload = compressed
    elif entry.compression == 8:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        try:
            payload = decompressor.decompress(compressed)
            payload += decompressor.flush()
        except zlib.error as exc:
            raise FetchError(f"DEFLATE failure for {entry.name!r}: {exc}") from exc
        if (
            not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise FetchError(f"invalid DEFLATE stream boundaries for {entry.name!r}")
    else:
        raise FetchError(
            f"unsupported compression method {entry.compression} for {entry.name!r}"
        )
    if len(payload) != entry.uncompressed_size:
        raise FetchError(
            f"uncompressed size disagreement for {entry.name!r}: "
            f"expected {entry.uncompressed_size}, received {len(payload)}"
        )
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    if checksum != entry.crc32:
        raise FetchError(
            f"CRC-32 disagreement for {entry.name!r}: "
            f"expected {entry.crc32:08x}, received {checksum:08x}"
        )
    return payload


def _write_atomic(destination: Path, payload: bytes) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass
        raise FetchError(f"cannot write {destination}: {exc}") from exc


def execute_plans(reader: RangeReader, plans: Sequence[DestinationPlan]) -> None:
    for index, plan in enumerate(plans, start=1):
        if plan.status == "verified":
            if (
                not plan.destination.is_file()
                or plan.destination.is_symlink()
                or _file_crc32(plan.destination)
                != (plan.entry.uncompressed_size, plan.entry.crc32)
            ):
                raise FetchError(
                    f"verified destination changed before extraction: {plan.destination}"
                )
            print(f"[{index:02d}/{len(plans):02d}] verified {plan.destination}")
            continue
        if plan.status not in ("download", "replace"):
            raise FetchError(f"cannot execute destination status {plan.status!r}")
        print(
            f"[{index:02d}/{len(plans):02d}] {plan.status} "
            f"{plan.entry.name} ({plan.entry.uncompressed_size} bytes)"
        )
        payload = read_member_payload(reader, plan.entry)
        if plan.destination.is_symlink():
            raise FetchError(
                f"refusing symbolic-link destination created during extraction: "
                f"{plan.destination}"
            )
        if plan.status == "download" and plan.destination.exists():
            if not plan.destination.is_file():
                raise FetchError(
                    f"destination appeared and is not a file: {plan.destination}"
                )
            if _file_crc32(plan.destination) == (
                plan.entry.uncompressed_size,
                plan.entry.crc32,
            ):
                print(f"[{index:02d}/{len(plans):02d}] verified {plan.destination}")
                continue
            raise FetchError(
                f"mismatched destination appeared during extraction: {plan.destination}"
            )
        _write_atomic(plan.destination, payload)
        written_size, written_crc = _file_crc32(plan.destination)
        if (written_size, written_crc) != (
            plan.entry.uncompressed_size,
            plan.entry.crc32,
        ):
            raise FetchError(f"post-write integrity check failed for {plan.destination}")


def verify_full_remote_md5(
    metadata: ArchiveMetadata,
    *,
    timeout: float,
    retries: int,
) -> None:
    """Stream the complete archive and compare its bytes with the pinned MD5."""
    request = Request(
        metadata.url,
        headers={"Accept-Encoding": "identity", "User-Agent": USER_AGENT},
    )
    response = _open_with_retries(request, timeout=timeout, retries=retries)
    digest = hashlib.md5(usedforsecurity=False)
    received = 0
    try:
        status_code = getattr(response, "status", response.getcode())
        _validate_zenodo_url(response.geturl(), "archive response URL")
        if status_code != 200:
            raise FetchError(f"full archive request returned HTTP {status_code}")
        declared = response.headers.get("Content-Length")
        if declared is None or not declared.isdigit() or int(declared) != metadata.size:
            raise FetchError(
                f"full archive Content-Length disagreement: {declared!r}"
            )
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            received += len(block)
            if received > metadata.size:
                raise FetchError("full archive response exceeded its declared size")
            digest.update(block)
    finally:
        response.close()
    if received != metadata.size:
        raise FetchError(
            f"full archive length disagreement: expected {metadata.size}, "
            f"received {received}"
        )
    actual = digest.hexdigest()
    if actual != metadata.md5:
        raise FetchError(
            f"full archive MD5 disagreement: expected {metadata.md5}, received {actual}"
        )


def _print_entry_list(selected: Sequence[ZipEntry]) -> None:
    print("member\tuncompressed_bytes\tcompressed_bytes\tcrc32\tmethod")
    for entry in selected:
        print(
            f"{entry.name}\t{entry.uncompressed_size}\t{entry.compressed_size}"
            f"\t{entry.crc32:08x}\t{entry.compression}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Range-fetch the declared left-atrial surface subset from the pinned "
            "Zenodo data.zip archive."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"destination root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the selected central-directory entries and exit without writing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check metadata, ZIP index, and local target states without writing",
    )
    parser.add_argument(
        "--overwrite-mismatched",
        action="store_true",
        help="replace an existing target only when its size or CRC-32 is wrong",
    )
    parser.add_argument(
        "--verify-full-archive-md5",
        action="store_true",
        help=(
            "stream all 1.6 GB before subset extraction to verify the archive MD5 "
            "from bytes, not only from pinned Zenodo API metadata"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"retries for transient requests (default: {DEFAULT_RETRIES})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.timeout <= 0:
        raise FetchError("--timeout must be positive")
    if args.retries < 0:
        raise FetchError("--retries cannot be negative")
    if args.list and args.verify_full_archive_md5:
        raise FetchError("--list cannot be combined with --verify-full-archive-md5")

    metadata = fetch_record_metadata(timeout=args.timeout, retries=args.retries)
    print(
        f"Pinned Zenodo archive metadata verified: {ARCHIVE_KEY}, "
        f"size={metadata.size}, md5={metadata.md5}"
    )
    if args.verify_full_archive_md5:
        print("Streaming the complete archive for byte-level MD5 verification...")
        verify_full_remote_md5(
            metadata, timeout=args.timeout, retries=args.retries
        )
        print("Full archive byte-level MD5 verified.")

    reader = HTTPRangeReader(
        metadata.url,
        metadata.size,
        timeout=args.timeout,
        retries=args.retries,
    )
    selected = select_declared_entries(read_central_directory(reader))
    if args.list:
        _print_entry_list(selected)
        return 0

    plans = plan_destinations(
        selected,
        args.output_root,
        overwrite_mismatched=args.overwrite_mismatched,
        fail_on_conflict=not args.dry_run,
    )
    if args.dry_run:
        for plan in plans:
            print(f"{plan.status}\t{plan.destination}\t{plan.entry.name}")
        print("Dry run complete; no files were written.")
        return 0

    execute_plans(reader, plans)
    print(
        f"Complete: {len(plans)} files are present and CRC-verified below "
        f"{args.output_root.expanduser().resolve(strict=False)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
