"""Synthetic verification driver for the triangular-surface P1 backend.

This command never reads cohort or patient files.  It writes a planar exact
case, a polygonal-annulus refinement table, and a curved-sphere exact-state
refinement table to a user-selected directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from surface_fem import assemble_p1, solve_labelled_capacity
from surface_mesh import (
    TriangleMesh,
    annular_tri_mesh,
    icosphere_tri_mesh,
    rectangular_tri_mesh,
)
from surface_phase import SurfacePhaseParameters, screened_reconstruction, solve_surface_phase


ROOT = Path(__file__).resolve().parents[1]


def _sparse_frobenius(matrix) -> float:
    return float(np.sqrt(np.sum(np.asarray(matrix.data, dtype=float) ** 2)))


def planar_exact_case(nx: int = 24, ny: int = 12) -> dict[str, object]:
    length, width, coefficient = 2.0, 1.0, 2.75
    mesh = rectangular_tri_mesh(nx, ny, length=length, width=width)
    operators = assemble_p1(mesh, coefficient)
    result = solve_labelled_capacity(operators, "boundary_id", 1, 2)
    exact = 1.0 - mesh.points[:, 0] / length
    error = result.potential - exact
    exact_capacity = coefficient * width / length

    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle) / np.sqrt(2.0), np.cos(angle) / np.sqrt(2.0), -1.0 / np.sqrt(2.0)),
            (np.sin(angle) / np.sqrt(2.0), np.cos(angle) / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
        )
    )
    rotated_mesh = TriangleMesh(
        mesh.points @ rotation.T,
        mesh.triangles,
        mesh.point_data,
        mesh.cell_data,
    )
    rotated = assemble_p1(rotated_mesh, coefficient)
    stiffness_difference = operators.stiffness - rotated.stiffness
    rigid_motion_defect = _sparse_frobenius(stiffness_difference) / max(
        _sparse_frobenius(operators.stiffness), np.finfo(float).tiny
    )
    constants = np.ones(mesh.n_vertices)
    row_sum_defect = float(
        np.linalg.norm(operators.stiffness @ constants, ord=np.inf)
        / max(_sparse_frobenius(operators.stiffness), np.finfo(float).tiny)
    )
    mass_area_defect = abs(float(constants @ (operators.mass @ constants)) - length * width) / (
        length * width
    )
    return {
        "source": "synthetic_planar_rectangle",
        "n_vertices": mesh.n_vertices,
        "n_triangles": mesh.n_triangles,
        "coefficient": coefficient,
        "exact_capacity": exact_capacity,
        "computed_capacity": result.value,
        "capacity_relative_error": abs(result.value - exact_capacity) / exact_capacity,
        "normalized_capacity": result.normalized_value,
        "normalized_capacity_error": abs(result.normalized_value - coefficient) / coefficient,
        "potential_Linf_error": float(np.max(np.abs(error))),
        "potential_L2_error": operators.mass_norm(error) / np.sqrt(operators.total_area),
        "free_relative_residual": result.relative_residual,
        "flux_energy_defect": result.flux_energy_defect,
        "constant_nullspace_relative_defect": row_sum_defect,
        "mass_area_relative_defect": mass_area_defect,
        "rigid_motion_relative_defect": rigid_motion_defect,
        "quality": asdict(operators.quality),
    }


def annulus_refinement(levels: Sequence[int]) -> list[dict[str, float | int | str]]:
    inner_radius, outer_radius, coefficient = 1.0, 2.0, 1.7
    exact_capacity = 2.0 * np.pi * coefficient / np.log(outer_radius / inner_radius)
    rows: list[dict[str, float | int | str]] = []
    previous_error: float | None = None
    previous_h: float | None = None
    for n_radial in levels:
        n_angular = 8 * n_radial
        mesh = annular_tri_mesh(
            n_radial,
            n_angular,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
        )
        operators = assemble_p1(mesh, coefficient)
        result = solve_labelled_capacity(operators, "boundary_id", 1, 2)
        relative_error = abs(result.value - exact_capacity) / exact_capacity
        h_proxy = 1.0 / n_radial
        order = 0.0
        order_defined = 0
        if previous_error is not None and previous_h is not None:
            order = float(np.log(previous_error / relative_error) / np.log(previous_h / h_proxy))
            order_defined = 1
        rows.append(
            {
                "source": "synthetic_polygonal_annulus",
                "n_radial": n_radial,
                "n_angular": n_angular,
                "n_vertices": mesh.n_vertices,
                "n_triangles": mesh.n_triangles,
                "h_proxy": h_proxy,
                "computed_capacity": result.value,
                "exact_circular_capacity": exact_capacity,
                "capacity_relative_error": relative_error,
                "observed_order": order,
                "order_defined": order_defined,
                "normalized_capacity": result.normalized_value,
                "free_relative_residual": result.relative_residual,
                "flux_energy_defect": result.flux_energy_defect,
                "minimum_angle_degrees": operators.quality.minimum_angle_degrees,
                "minimum_mean_ratio_quality": operators.quality.minimum_mean_ratio_quality,
            }
        )
        previous_error, previous_h = relative_error, h_proxy
    return rows


def sphere_screened_refinement(levels: Sequence[int]) -> list[dict[str, float | int | str]]:
    """Verify a screened solve using the degree-one spherical harmonic ``x``."""

    radius = 1.0
    epsilon = 0.2
    confidence_value = 1.0
    length_scale = 0.5
    rows: list[dict[str, float | int | str]] = []
    previous_error: float | None = None
    previous_h: float | None = None
    for level in levels:
        mesh = icosphere_tri_mesh(level, radius=radius)
        operators = assemble_p1(mesh)
        exact = mesh.points[:, 0] / radius
        confidence = np.full(mesh.n_vertices, confidence_value)
        # On the unit sphere, -Delta_S x = 2x.  This prescribed forcing
        # verifies the curved operator and solve, rather than an observation
        # compatibility condition.
        forcing = (
            epsilon + confidence_value + 2.0 * length_scale**2 / radius**2
        ) * exact
        result = screened_reconstruction(
            operators,
            confidence,
            forcing,
            epsilon=epsilon,
            length_scale=length_scale,
            enforce_data_compatibility=False,
        )
        error = result.state - exact
        relative_l2_error = operators.mass_norm(error) / operators.mass_norm(exact)
        h_proxy = 2.0 ** (-level)
        order = 0.0
        order_defined = 0
        if previous_error is not None and previous_h is not None:
            order = float(
                np.log(previous_error / relative_l2_error)
                / np.log(previous_h / h_proxy)
            )
            order_defined = 1
        rows.append(
            {
                "source": "synthetic_unit_icosphere",
                "refinement_level": level,
                "n_vertices": mesh.n_vertices,
                "n_triangles": mesh.n_triangles,
                "h_proxy": h_proxy,
                "relative_L2_error": relative_l2_error,
                "observed_order": order,
                "order_defined": order_defined,
                "strong_relative_residual": result.relative_residual,
                "surface_area": operators.total_area,
                "surface_area_relative_error": abs(operators.total_area - 4.0 * np.pi)
                / (4.0 * np.pi),
            }
        )
        previous_error, previous_h = relative_l2_error, h_proxy
    return rows


def sphere_graph_refinement(levels: Sequence[int]) -> list[dict[str, float | int | str]]:
    """Check direct-graph residuals and nested-node stability on an icosphere."""

    rows: list[dict[str, float | int | str]] = []
    previous_state: np.ndarray | None = None
    parameters = SurfacePhaseParameters.graph(
        mu=0.30,
        nu=0.20,
        dt=0.03,
        admm_tolerance=1.0e-7,
        admm_max_iterations=8000,
    )
    for level in levels:
        mesh = icosphere_tri_mesh(level)
        operators = assemble_p1(mesh)
        datum = np.clip(0.65 * mesh.points[:, 0] - 0.20 * mesh.points[:, 2], -1.0, 1.0)
        confidence = np.ones(mesh.n_vertices)
        state, _ = solve_surface_phase(
            operators,
            datum,
            confidence,
            confidence * datum,
            parameters,
            nsteps=4,
        )
        nested_error = 0.0
        nested_error_defined = 0
        if previous_state is not None:
            # Icosahedral subdivision retains the preceding vertices first.
            nested_error = float(
                np.sqrt(np.mean((state.u[: previous_state.size] - previous_state) ** 2))
            )
            nested_error_defined = 1
        rows.append(
            {
                "source": "synthetic_unit_icosphere",
                "refinement_level": level,
                "n_vertices": mesh.n_vertices,
                "n_triangles": mesh.n_triangles,
                "pseudo_time_steps": 4,
                "nested_node_RMSE": nested_error,
                "nested_error_defined": nested_error_defined,
                "state_min": float(np.min(state.u)),
                "state_max": float(np.max(state.u)),
                "max_admm_iterations": int(
                    max(item["admm_iterations"] for item in state.diagnostics)
                ),
                "max_state_relative_residual": float(
                    max(item["state_relative_residual"] for item in state.diagnostics)
                ),
                "max_graph_projection_relative_residual": float(
                    max(
                        item["graph_projection_relative_residual"]
                        for item in state.diagnostics
                    )
                ),
                "max_mass_source_defect": float(
                    max(item["mass_source_defect"] for item in state.diagnostics)
                ),
            }
        )
        previous_state = state.u.copy()
    return rows


def verify_results(
    planar: dict[str, object],
    annulus: list[dict[str, object]],
    sphere: list[dict[str, object]],
    sphere_graph: list[dict[str, object]],
) -> None:
    for key, tolerance in {
        "capacity_relative_error": 2.0e-12,
        "normalized_capacity_error": 2.0e-12,
        "potential_Linf_error": 2.0e-12,
        "potential_L2_error": 2.0e-12,
        "free_relative_residual": 2.0e-12,
        "flux_energy_defect": 2.0e-11,
        "constant_nullspace_relative_defect": 2.0e-14,
        "mass_area_relative_defect": 2.0e-14,
        "rigid_motion_relative_defect": 2.0e-14,
    }.items():
        if float(planar[key]) > tolerance:
            raise AssertionError(f"planar {key}={planar[key]:.3e} exceeds {tolerance:.3e}")
    errors = np.asarray([float(row["capacity_relative_error"]) for row in annulus])
    if not np.all(np.diff(errors) < 0.0):
        raise AssertionError("annulus capacity error is not strictly decreasing")
    orders = np.asarray([float(row["observed_order"]) for row in annulus[1:]])
    if not np.all(np.isfinite(orders)) or float(np.min(orders)) < 1.8:
        raise AssertionError("annulus capacity does not show the expected second-order trend")
    if errors[-1] > 5.0e-4:
        raise AssertionError("finest annulus capacity error exceeds 5e-4")
    if max(float(row["free_relative_residual"]) for row in annulus) > 5.0e-12:
        raise AssertionError("an annulus linear solve residual is too large")
    if max(float(row["flux_energy_defect"]) for row in annulus) > 5.0e-11:
        raise AssertionError("an annulus flux/energy identity does not close")
    sphere_errors = np.asarray([float(row["relative_L2_error"]) for row in sphere])
    if not np.all(np.diff(sphere_errors) < 0.0):
        raise AssertionError("curved screened-state error is not strictly decreasing")
    sphere_orders = np.asarray([float(row["observed_order"]) for row in sphere[1:]])
    if not np.all(np.isfinite(sphere_orders)) or float(np.min(sphere_orders[-2:])) < 1.8:
        raise AssertionError("curved screened solve does not approach second order")
    if sphere_errors[-1] > 8.0e-5:
        raise AssertionError("finest curved screened-state error exceeds 8e-5")
    if max(float(row["strong_relative_residual"]) for row in sphere) > 2.0e-12:
        raise AssertionError("a curved screened solve residual is too large")
    graph_errors = np.asarray(
        [float(row["nested_node_RMSE"]) for row in sphere_graph[1:]]
    )
    if not np.all(np.diff(graph_errors) < 0.0):
        raise AssertionError("curved graph states are not stabilising under refinement")
    if max(float(row["max_state_relative_residual"]) for row in sphere_graph) > 1.1e-7:
        raise AssertionError("a curved graph state residual exceeds tolerance")
    if max(
        float(row["max_graph_projection_relative_residual"]) for row in sphere_graph
    ) > 1.0e-8:
        raise AssertionError("a curved graph projection residual exceeds tolerance")
    if max(float(row["max_mass_source_defect"]) for row in sphere_graph) > 1.0e-8:
        raise AssertionError("a curved graph mass identity does not close")


def _parse_levels(text: str) -> list[int]:
    try:
        levels = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from exc
    if len(levels) < 2 or any(level < 2 for level in levels):
        raise argparse.ArgumentTypeError("provide at least two radial levels, each >= 2")
    if any(right <= left for left, right in zip(levels, levels[1:])):
        raise argparse.ArgumentTypeError("radial levels must be strictly increasing")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / "surface_verification",
        help="directory for the two synthetic verification files",
    )
    parser.add_argument(
        "--annulus-levels",
        type=_parse_levels,
        default=_parse_levels("4,8,16"),
        help="comma-separated radial refinements (default: 4,8,16)",
    )
    args = parser.parse_args()

    planar = planar_exact_case()
    annulus = annulus_refinement(args.annulus_levels)
    sphere = sphere_screened_refinement((1, 2, 3, 4))
    sphere_graph = sphere_graph_refinement((1, 2, 3))
    verify_results(planar, annulus, sphere, sphere_graph)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    planar_path = args.output_dir / "surface_planar_exact.json"
    annulus_path = args.output_dir / "surface_annulus_refinement.csv"
    sphere_path = args.output_dir / "surface_sphere_screened_refinement.csv"
    graph_path = args.output_dir / "surface_sphere_graph_refinement.csv"
    planar_path.write_text(json.dumps(planar, indent=2, sort_keys=True) + "\n")
    with annulus_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annulus[0]))
        writer.writeheader()
        writer.writerows(annulus)
    with sphere_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sphere[0]))
        writer.writeheader()
        writer.writerows(sphere)
    with graph_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sphere_graph[0]))
        writer.writeheader()
        writer.writerows(sphere_graph)

    print(
        "surface verification passed: "
        f"planar capacity rel. error={float(planar['capacity_relative_error']):.3e}; "
        f"annulus finest rel. error={float(annulus[-1]['capacity_relative_error']):.3e}; "
        f"sphere finest rel. L2 error={float(sphere[-1]['relative_L2_error']):.3e}"
    )
    print(planar_path)
    print(annulus_path)
    print(sphere_path)
    print(graph_path)


if __name__ == "__main__":
    main()
