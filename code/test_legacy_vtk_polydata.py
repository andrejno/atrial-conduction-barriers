"""Checks for the strict legacy-VTK POLYDATA reader."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from legacy_vtk_polydata import clean_triangular_surface, read_legacy_polydata


class LegacyPolyDataTests(unittest.TestCase):
    def test_cleanup_removes_degenerate_and_duplicate_cells_without_vertex_merging(self) -> None:
        points = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0 + 1.0e-12, 0.0),  # deliberately near, but distinct
                (0.5, 0.0, 0.0),  # makes triangle 3 collinear
                (9.0, 9.0, 9.0),  # isolated
            )
        )
        triangles = np.asarray(
            (
                (0, 1, 2),
                (2, 1, 0),  # duplicate of triangle 0
                (0, 1, 3),  # geometrically near triangle 0, but not merged
                (0, 1, 4),  # zero area
            ),
            dtype=np.int64,
        )
        point_data = {
            "point_id": np.arange(6, dtype=np.int32),
            "vector": np.arange(18, dtype=np.float32).reshape(6, 3),
        }
        cell_data = {
            "source_polygon_id": np.asarray((10, 11, 12, 13), dtype=np.int64),
            "value": np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float32),
        }
        clean_points, clean_cells, clean_pd, clean_cd, report = clean_triangular_surface(
            points, triangles, point_data, cell_data
        )

        self.assertEqual(clean_points.shape, (4, 3))
        np.testing.assert_array_equal(clean_pd["point_id"], (0, 1, 2, 3))
        self.assertEqual(clean_pd["vector"].dtype, np.dtype("float32"))
        np.testing.assert_array_equal(clean_cells, ((0, 1, 2), (0, 1, 3)))
        np.testing.assert_array_equal(clean_cd["value"], (1.0, 3.0))
        self.assertEqual(report.removed_zero_area_triangle_indices, (3,))
        self.assertEqual(report.removed_duplicate_triangle_indices, (1,))
        self.assertEqual(report.removed_source_polygon_ids, (11, 13))
        self.assertEqual(report.removed_isolated_point_indices, (4, 5))
        # The two extremely close points remain distinct and separately indexed.
        self.assertGreater(np.linalg.norm(clean_points[2] - clean_points[3]), 0.0)

    def test_ascii_classic_fields_and_shorter_quad_diagonal(self) -> None:
        text = """# vtk DataFile Version 4.0
test mesh
ASCII
DATASET POLYDATA
POINTS 4 double
0 0 0  1 0 0  2 1 0  0 1 0
POLYGONS 1 5
4 0 1 2 3
CELL_DATA 1
FIELD FieldData 1
label 1 1 int
7
POINT_DATA 4
FIELD FieldData 2
score 1 4 float
0.1 0.2 0.3 0.4
direction 3 4 double
1 0 0  0 1 0  0 0 1  1 1 1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classic.vtk"
            path.write_text(text, encoding="ascii")
            points, triangles, point_data, cell_data = read_legacy_polydata(path)

        # The 1--3 diagonal is shorter than 0--2.
        np.testing.assert_array_equal(triangles, ((0, 1, 3), (1, 2, 3)))
        self.assertEqual(points.dtype, np.dtype("float64"))
        self.assertEqual(point_data["score"].dtype, np.dtype("float32"))
        self.assertEqual(point_data["direction"].shape, (4, 3))
        np.testing.assert_array_equal(cell_data["label"], (7, 7))
        np.testing.assert_array_equal(cell_data["source_polygon_id"], (0, 0))
        np.testing.assert_array_equal(cell_data["source_polygon_size"], (4, 4))

    def test_binary_offset_connectivity_and_native_dtypes(self) -> None:
        points = np.asarray(
            ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=">f8"
        )
        offsets = np.asarray((0, 4), dtype=">i8")
        connectivity = np.asarray((0, 1, 2, 3), dtype=">i8")
        score = np.asarray((1, 2, 3, 4), dtype=">i4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offsets.vtk"
            with path.open("wb") as stream:
                stream.write(b"# vtk DataFile Version 5.1\ntest mesh\nBINARY\n")
                stream.write(b"DATASET POLYDATA\nPOINTS 4 double\n")
                stream.write(points.tobytes() + b"\n")
                # VTK 5.1 records the number of offsets, not the number of cells.
                stream.write(b"POLYGONS 2 4\nOFFSETS vtktypeint64\n")
                stream.write(offsets.tobytes() + b"\n")
                stream.write(b"CONNECTIVITY vtktypeint64\n")
                stream.write(connectivity.tobytes() + b"\n")
                stream.write(b"POINT_DATA 4\nFIELD FieldData 1\nscore 1 4 int\n")
                stream.write(score.tobytes() + b"\n")
            loaded_points, triangles, point_data, cell_data = read_legacy_polydata(path)

        self.assertTrue(loaded_points.dtype.isnative)
        self.assertTrue(point_data["score"].dtype.isnative)
        np.testing.assert_array_equal(point_data["score"], (1, 2, 3, 4))
        # Equal diagonal lengths use the lexicographically smaller (0, 2) pair.
        np.testing.assert_array_equal(triangles, ((0, 1, 2), (0, 2, 3)))
        np.testing.assert_array_equal(cell_data["source_polygon_id"], (0, 0))

    def test_all_extracted_zenodo_meshes_when_available(self) -> None:
        configured = os.environ.get("ZENODO_ERP_MESH_ROOT")
        candidates = [
            Path(configured) if configured else Path("__not_configured__"),
            CODE.parent.parent.parent / "external_data" / "zenodo_erp" / "meshes",
        ]
        root = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if root is None:
            self.skipTest("set ZENODO_ERP_MESH_ROOT to run the seven-mesh integration check")
        for patient in range(1, 8):
            path = root / f"P{patient}" / f"P{patient}_with_erp_lat_bi.vtk"
            self.assertTrue(path.is_file(), str(path))
            with self.subTest(patient=patient):
                points, triangles, point_data, cell_data = read_legacy_polydata(path)
                self.assertEqual(points.ndim, 2)
                self.assertEqual(points.shape[1], 3)
                self.assertEqual(triangles.ndim, 2)
                self.assertEqual(triangles.shape[1], 3)
                self.assertEqual(point_data["bi"].shape, (points.shape[0],))
                self.assertEqual(point_data["lat"].shape, (points.shape[0],))
                self.assertEqual(point_data["erp_laplace"].shape, (points.shape[0],))
                self.assertEqual(
                    cell_data["source_polygon_id"].shape, (triangles.shape[0],)
                )
                self.assertEqual(
                    cell_data["source_polygon_size"].shape, (triangles.shape[0],)
                )
                self.assertTrue(
                    set(np.unique(cell_data["source_polygon_size"])).issubset({3, 4})
                )
                clean = clean_triangular_surface(
                    points, triangles, point_data, cell_data
                )
                clean_points, clean_triangles, clean_pd, clean_cd, report = clean
                self.assertEqual(clean_points.shape[1], 3)
                self.assertEqual(clean_triangles.shape[1], 3)
                self.assertEqual(clean_pd["bi"].shape[0], clean_points.shape[0])
                self.assertEqual(
                    clean_cd["source_polygon_id"].shape[0], clean_triangles.shape[0]
                )
                if patient == 4:
                    self.assertEqual(report.removed_zero_area_triangle_indices, (7386,))
                    self.assertEqual(report.removed_source_polygon_ids, (7132,))
                else:
                    self.assertEqual(report.removed_zero_area_triangle_indices, ())


if __name__ == "__main__":
    unittest.main()
