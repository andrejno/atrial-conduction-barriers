"""Unit and manufactured-solution checks for the surface P1 backend."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from surface_fem import assemble_p1, solve_dirichlet_capacity, solve_labelled_capacity
from surface_mesh import (
    TriangleMesh,
    _tetrahedral_boundary,
    annular_tri_mesh,
    icosphere_tri_mesh,
    inspect_mesh_file,
    rectangular_tri_mesh,
)

try:
    import meshio as _meshio
except ImportError:  # The VTK dependency is intentionally optional.
    _meshio = None


class SurfaceMeshTests(unittest.TestCase):
    def test_icosphere_is_closed_oriented_and_refines(self) -> None:
        coarse = icosphere_tri_mesh(1)
        fine = icosphere_tri_mesh(2)
        for mesh in (coarse, fine):
            report = mesh.assert_valid(
                require_connected=True, require_consistent_orientation=True
            )
            self.assertEqual(report.boundary_edges, 0)
            self.assertEqual(report.euler_characteristic, 2)
            self.assertTrue(np.allclose(np.linalg.norm(mesh.points, axis=1), 1.0))
        self.assertEqual(fine.n_triangles, 4 * coarse.n_triangles)

    def test_rectangle_geometry_qc(self) -> None:
        mesh = rectangular_tri_mesh(8, 5, length=2.0, width=1.0)
        report = mesh.assert_valid(require_connected=True, require_consistent_orientation=True)
        self.assertTrue(report.valid_for_p1)
        self.assertEqual(report.euler_characteristic, 1)
        self.assertEqual(report.boundary_components, 1)
        self.assertEqual(report.boundary_branch_vertices, 0)
        self.assertAlmostEqual(report.total_area, 2.0, places=14)

    def test_invalid_nonmanifold_mesh_is_rejected(self) -> None:
        points = np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)),
            dtype=float,
        )
        mesh = TriangleMesh(points, ((0, 1, 2), (1, 0, 3), (0, 1, 4)))
        self.assertEqual(mesh.quality_report().nonmanifold_edges, 1)
        with self.assertRaisesRegex(ValueError, "non-manifold"):
            assemble_p1(mesh)

    def test_disconnected_vertex_link_is_rejected(self) -> None:
        points = np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        )
        mesh = TriangleMesh(points, ((0, 1, 2), (0, 3, 4)))
        self.assertEqual(mesh.quality_report().nonmanifold_vertices, 1)
        with self.assertRaisesRegex(ValueError, "vertex links"):
            mesh.assert_valid()

    def test_tetrahedral_boundary_is_closed_and_outward(self) -> None:
        points = np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.2, 0.2)),
            dtype=float,
        )
        faces, parents, original_ids = _tetrahedral_boundary(points, np.asarray(((0, 1, 2, 3),)))
        mesh = TriangleMesh(points[original_ids], faces)
        report = mesh.assert_valid(
            require_connected=True, require_consistent_orientation=True
        )
        self.assertEqual(report.euler_characteristic, 2)
        self.assertEqual(report.boundary_edges, 0)
        self.assertEqual(set(parents.tolist()), {0})
        self.assertNotIn(4, original_ids.tolist())
        centroid = np.mean(mesh.points, axis=0)
        for face in mesh.triangles:
            xyz = mesh.points[face]
            normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
            self.assertGreater(float(np.dot(normal, np.mean(xyz, axis=0) - centroid)), 0.0)

    @unittest.skipUnless(_meshio is not None, "optional meshio package is not installed")
    def test_legacy_vtk_tetrahedron_round_trip(self) -> None:
        assert _meshio is not None
        points = np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float
        )
        tetrahedra = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
        labels = np.asarray((1, 2, 2, 1), dtype=np.int16)
        material = np.asarray((0.35,), dtype=float)
        raw = _meshio.Mesh(
            points,
            [("tetra", tetrahedra)],
            point_data={"electrode": labels},
            cell_data={"material": [material]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_tetra.vtk"
            _meshio.write(path, raw, file_format="vtk", binary=False)
            inventory = inspect_mesh_file(path)
            loaded = TriangleMesh.from_file(path, cell_source="tetra_boundary")
        self.assertEqual(inventory["n_points"], 4)
        self.assertEqual(inventory["cell_blocks"][0]["type"], "tetra")
        self.assertEqual(loaded.n_vertices, 4)
        self.assertEqual(loaded.n_triangles, 4)
        np.testing.assert_array_equal(loaded.point_data["electrode"], labels)
        np.testing.assert_allclose(loaded.cell_data["material"], 0.35)
        np.testing.assert_array_equal(loaded.cell_data["parent_tetrahedron"], 0)
        loaded.assert_valid(require_connected=True, require_consistent_orientation=True)


class SurfaceFemTests(unittest.TestCase):
    def test_planar_affine_solution_and_capacity_are_exact(self) -> None:
        length, width, coefficient = 2.0, 1.0, 2.5
        mesh = rectangular_tri_mesh(12, 7, length=length, width=width)
        operators = assemble_p1(mesh, coefficient)
        result = solve_labelled_capacity(operators, "boundary_id", 1, 2)
        exact = 1.0 - mesh.points[:, 0] / length
        exact_capacity = coefficient * width / length
        self.assertLess(float(np.max(np.abs(result.potential - exact))), 2.0e-14)
        self.assertLess(abs(result.value - exact_capacity) / exact_capacity, 2.0e-14)
        self.assertLess(abs(result.normalized_value - coefficient) / coefficient, 2.0e-14)
        self.assertLess(result.relative_residual, 2.0e-14)
        self.assertLess(result.flux_energy_defect, 5.0e-14)

    def test_mass_and_weighted_operator_identities(self) -> None:
        mesh = rectangular_tri_mesh(7, 4, length=1.4, width=0.8)
        coefficient = 0.5 + mesh.points[:, 0]
        operators = assemble_p1(mesh, coefficient, coefficient_location="node")
        ones = np.ones(mesh.n_vertices)
        self.assertLess(float(np.linalg.norm(operators.stiffness @ ones, ord=np.inf)), 2.0e-14)
        self.assertLess(float(np.linalg.norm(operators.apply_laplacian(ones))), 2.0e-12)
        self.assertAlmostEqual(operators.mass_inner(ones, ones), 1.4 * 0.8, places=13)
        self.assertAlmostEqual(operators.mass_mean(ones), 1.0, places=14)
        biharmonic = operators.lumped_biharmonic_matrix()
        symmetry = biharmonic - biharmonic.T
        self.assertLess(float(np.linalg.norm(symmetry.data)), 2.0e-12)
        trial = np.sin(mesh.points[:, 0]) + 0.3 * mesh.points[:, 1]
        self.assertGreaterEqual(float(trial @ (biharmonic @ trial)), -1.0e-11)

    def test_capacity_scales_with_constant_coefficient(self) -> None:
        mesh = rectangular_tri_mesh(9, 5)
        source = mesh.labelled_nodes("boundary_id", 1)
        sink = mesh.labelled_nodes("boundary_id", 2)
        values = []
        for coefficient in (0.2, 1.0, 3.4):
            result = solve_dirichlet_capacity(
                assemble_p1(mesh, coefficient), source, sink
            )
            values.append(result.value)
            self.assertAlmostEqual(result.normalized_value, coefficient, places=13)
        self.assertTrue(np.all(np.diff(values) > 0.0))

    def test_polygonal_annulus_capacity_converges_quadratically(self) -> None:
        coefficient = 1.7
        exact = 2.0 * math.pi * coefficient / math.log(2.0)
        errors: list[float] = []
        for n_radial in (4, 8, 16):
            mesh = annular_tri_mesh(n_radial, 8 * n_radial)
            result = solve_labelled_capacity(
                assemble_p1(mesh, coefficient), "boundary_id", 1, 2
            )
            errors.append(abs(result.value - exact) / exact)
            self.assertLess(result.relative_residual, 5.0e-12)
            self.assertLess(result.flux_energy_defect, 5.0e-11)
        self.assertTrue(np.all(np.diff(errors) < 0.0))
        orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
        self.assertGreater(float(np.min(orders)), 1.8)
        self.assertLess(errors[-1], 5.0e-4)

    def test_unanchored_component_is_rejected(self) -> None:
        left = rectangular_tri_mesh(2, 2)
        right_points = left.points + np.asarray((3.0, 0.0, 0.0))
        points = np.vstack((left.points, right_points))
        triangles = np.vstack((left.triangles, left.triangles + left.n_vertices))
        mesh = TriangleMesh(points, triangles)
        operators = assemble_p1(mesh)
        source = np.flatnonzero(np.isclose(left.points[:, 0], 0.0))
        sink = np.flatnonzero(np.isclose(left.points[:, 0], 2.0))
        with self.assertRaisesRegex(ValueError, "no Dirichlet vertex"):
            solve_dirichlet_capacity(operators, source, sink)


class SurfaceCliTests(unittest.TestCase):
    def test_cli_writes_only_declared_synthetic_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CODE / "run_surface_verification.py"),
                    "--output-dir",
                    directory,
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            names = sorted(path.name for path in Path(directory).iterdir())
            self.assertEqual(
                names,
                [
                    "surface_annulus_refinement.csv",
                    "surface_planar_exact.json",
                    "surface_sphere_graph_refinement.csv",
                    "surface_sphere_screened_refinement.csv",
                ],
            )
            self.assertIn("surface verification passed", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
