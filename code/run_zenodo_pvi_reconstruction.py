"""Patient-surface reconstruction on the open Zenodo prior-PVI EAM cohort.

This is a real-data replacement for the former synthetic end-to-end experiment.
It uses the six left-atrial surfaces in Zenodo record 10.5281/zenodo.10726677.
The source ``bi`` field is a bipolar-voltage map acquired during sinus rhythm in
patients with a history of AF and prior PVI.  It is not a direct measurement of
electrical block.  Following the source publication, 0.1 mV is used only as a
voltage-defined ablation-lesion surrogate.

Two geometry-only blackouts are made in each patient: one surrounding the two
left PVs and one surrounding the two right PVs.  Official bilayer element tags,
not coordinate orientation or voltage, identify the raw-surface boundary loops.
The continuous target

    b(V) = 2 ** (-V / 0.1 mV),     q = 2 b - 1,

is bounded and has q=0 exactly at 0.1 mV.  ``b`` is a threshold-anchored score,
not a probability.  Contact locations, masks, numerical parameters and horizon
are deterministic and independent of voltage values.  Blackout confidence and
forcing are explicitly reset to exact zero after kernel accumulation.

The six patients, rather than PVs or masks, are the inferential units.  The two
mask results and four hidden-PV capacity results are averaged within patient
before paired intervals and exact sign-flip tests are computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.collections import PolyCollection
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

try:  # Direct script and package-style imports are both supported.
    from legacy_vtk_polydata import clean_triangular_surface, read_legacy_polydata
    from patient_method_checkpoint import input_signature, load_completed, save_completed
    from run_patient_reconstruction import build_capacity_domain, capacity_from_score
    from surface_fem import assemble_p1
    from surface_mesh import TriangleMesh
    from surface_phase import (
        SurfacePhaseParameters,
        screened_reconstruction,
        solve_surface_phase,
    )
except ImportError:  # pragma: no cover
    from .legacy_vtk_polydata import clean_triangular_surface, read_legacy_polydata
    from .patient_method_checkpoint import input_signature, load_completed, save_completed
    from .run_patient_reconstruction import build_capacity_domain, capacity_from_score
    from .surface_fem import assemble_p1
    from .surface_mesh import TriangleMesh
    from .surface_phase import (
        SurfacePhaseParameters,
        screened_reconstruction,
        solve_surface_phase,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT_ROOT = ROOT / "external_data" / "zenodo_erp" / "meshes"
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
DEFAULT_CHECKPOINT_DIR = ROOT / "tmp" / "zenodo_pvi_method_checkpoints"

PATIENT_IDS = ("P1", "P3", "P4", "P5", "P6", "P7")
EXCLUDED_RIGHT_ATRIUM = "P2"
PV_TAGS = {
    "LSPV": 13,
    "LIPV": 14,
    "RSPV": 15,
    "RIPV": 16,
}
BOUNDARY_TAGS = {"MV": 12, **PV_TAGS}
MASK_GROUPS = {
    "left_pair": ("LSPV", "LIPV"),
    "right_pair": ("RSPV", "RIPV"),
}

# Locked from the strengthened synthetic reconstruction experiment wherever a
# direct analogue exists.  Coordinates remain in mm, hence reference length 1.
REFERENCE_LENGTH_MM = 1.0
VOLTAGE_THRESHOLD_MV = 0.1
CONTACT_VOXEL_SIZE_MM = 5.0
KERNEL_SUPPORT_MM = 2.25
KERNEL_WEIGHT = 15.0
EVALUATION_WIDTH_MM = 10.0
GUARD_WIDTH_MM = 2.0
BLACKOUT_WIDTH_MM = EVALUATION_WIDTH_MM + GUARD_WIDTH_MM
CAPACITY_WIDTH_MM = 10.0
SCREEN_EPSILON = 0.05
SCREEN_LENGTH_SCALE = 1.50
PHASE_MU = 0.30
GRAPH_NU = 0.20
PHASE_DT = 0.01
PHASE_HORIZON = 1.20
PHASE_STEPS = 120
TERMINAL_STATES = 12
RHO_FACTOR = 20.0
ADMM_TOLERANCE = 5.0e-6
ADMM_MAX_ITERATIONS = 2500
LINEAR_TOLERANCE = 1.0e-10
LINEAR_ABSOLUTE_TOLERANCE = 1.0e-12
LINEAR_MAX_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_RESAMPLES = 20000
DATASET_DOI = "10.5281/zenodo.10726677"
SOURCE_PAPER_DOI = "10.1093/europace/euae215"
LICENSE = "CC BY 4.0"

if not np.isclose(PHASE_DT * PHASE_STEPS, PHASE_HORIZON):
    raise RuntimeError("locked phase horizon and step count are inconsistent")
if TERMINAL_STATES > PHASE_STEPS:
    raise RuntimeError("terminal averaging window exceeds the locked phase horizon")


@dataclass(frozen=True)
class BoundaryAssignment:
    label: str
    official_tag: int
    raw_boundary_nodes: int
    raw_perimeter_mm: float
    median_distance_mm: float
    p95_distance_mm: float
    maximum_distance_mm: float
    nearest_tag_fraction: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def _read_carp_points(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"bilayer point file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline().strip()
    try:
        declared = int(first)
    except ValueError as exc:
        raise ValueError(f"invalid CARP point count in {path}: {first!r}") from exc
    points = np.loadtxt(path, dtype=float, skiprows=1)
    if points.shape != (declared, 3) or not np.isfinite(points).all():
        raise ValueError(
            f"CARP points in {path} have shape {points.shape}, expected {(declared, 3)}"
        )
    # Official CARP bilayer coordinates are micrometres; simple VTK coordinates
    # are millimetres.  The 1/1000 conversion is checked again by tag distances.
    return np.asarray(points / 1000.0, dtype=float)


def _read_tag_vertex_ids(path: Path, wanted_tags: Sequence[int]) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"bilayer element file not found: {path}")
    wanted = set(map(int, wanted_tags))
    selected: dict[int, set[int]] = {tag: set() for tag in wanted}
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline().strip()
        try:
            declared = int(first)
        except ValueError as exc:
            raise ValueError(f"invalid CARP element count in {path}: {first!r}") from exc
        received = 0
        for received, line in enumerate(stream, start=1):
            tokens = line.split()
            if tokens and tokens[0] == "Ln" and len(tokens) == 4:
                # Transmural connector elements in the bilayer do not define
                # surface regions and therefore do not enter boundary transfer.
                continue
            if len(tokens) != 5 or tokens[0] != "Tr":
                raise ValueError(f"unsupported CARP element on line {received + 1} of {path}")
            a, b, c, tag = map(int, tokens[1:])
            if tag in selected:
                selected[tag].update((a, b, c))
    if received != declared:
        raise ValueError(f"CARP element count {received} != declared {declared} in {path}")
    missing = [tag for tag, nodes in selected.items() if not nodes]
    if missing:
        raise ValueError(f"bilayer element file lacks required tags {missing}: {path}")
    return {
        tag: np.fromiter(sorted(nodes), dtype=np.int64, count=len(nodes))
        for tag, nodes in selected.items()
    }


def _boundary_cycles(mesh: TriangleMesh) -> list[np.ndarray]:
    edges = np.asarray(mesh.boundary_edges(), dtype=np.int64)
    if len(edges) == 0:
        raise ValueError("surface has no boundary cycles")
    degree = np.bincount(edges.ravel(), minlength=mesh.n_vertices)
    bad = np.flatnonzero((degree != 0) & (degree != 2))
    if len(bad):
        raise ValueError(f"surface boundary has {len(bad)} branch/end vertices")
    nodes = np.unique(edges.ravel())
    remap = np.full(mesh.n_vertices, -1, dtype=np.int64)
    remap[nodes] = np.arange(len(nodes))
    graph = coo_matrix(
        (
            np.ones(2 * len(edges), dtype=np.int8),
            (
                np.r_[remap[edges[:, 0]], remap[edges[:, 1]]],
                np.r_[remap[edges[:, 1]], remap[edges[:, 0]]],
            ),
        ),
        shape=(len(nodes), len(nodes)),
    ).tocsr()
    count, labels = connected_components(graph, directed=False, return_labels=True)
    cycles: list[np.ndarray] = []
    adjacency: dict[int, list[int]] = {int(node): [] for node in nodes}
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    for component in range(count):
        component_nodes = nodes[labels == component]
        start = int(np.min(component_nodes))
        first = min(adjacency[start])
        ordered = [start]
        previous, current = start, first
        while current != start:
            ordered.append(current)
            neighbours = adjacency[current]
            nxt = neighbours[0] if neighbours[1] == previous else neighbours[1]
            previous, current = current, nxt
            if len(ordered) > len(component_nodes):
                raise ValueError("boundary traversal did not close after one component")
        if len(ordered) != len(component_nodes):
            raise ValueError("boundary component is not one simple cycle")
        cycles.append(np.asarray(ordered, dtype=np.int64))
    return cycles


def _cycle_perimeter(points: np.ndarray, cycle: np.ndarray) -> float:
    closed = np.r_[cycle, cycle[0]]
    return float(np.linalg.norm(points[closed[1:]] - points[closed[:-1]], axis=1).sum())


def _assign_official_boundaries(
    patient_dir: Path,
    mesh: TriangleMesh,
) -> tuple[dict[str, np.ndarray], list[BoundaryAssignment]]:
    cycles = _boundary_cycles(mesh)
    if len(cycles) != len(BOUNDARY_TAGS):
        raise ValueError(
            f"expected five LA boundary cycles, found {len(cycles)} in {patient_dir}"
        )
    bilayer_dir = patient_dir / "bilayer"
    bilayer_points = _read_carp_points(bilayer_dir / "LA_bilayer_with_fiber_um.pts")
    tag_ids = _read_tag_vertex_ids(
        bilayer_dir / "LA_bilayer_with_fiber_um.elem", tuple(BOUNDARY_TAGS.values())
    )
    tag_trees = {
        tag: cKDTree(bilayer_points[indices]) for tag, indices in tag_ids.items()
    }
    labels = tuple(BOUNDARY_TAGS)
    distance_cube: list[list[np.ndarray]] = []
    costs = np.empty((len(cycles), len(labels)), dtype=float)
    for cycle_index, cycle in enumerate(cycles):
        row: list[np.ndarray] = []
        for label_index, label in enumerate(labels):
            distances = np.asarray(
                tag_trees[BOUNDARY_TAGS[label]].query(mesh.points[cycle], workers=1)[0],
                dtype=float,
            )
            row.append(distances)
            costs[cycle_index, label_index] = float(np.median(distances))
        distance_cube.append(row)
    cycle_indices, label_indices = linear_sum_assignment(costs)
    mapping: dict[str, np.ndarray] = {}
    records: list[BoundaryAssignment] = []
    for cycle_index, label_index in zip(cycle_indices, label_indices):
        label = labels[int(label_index)]
        cycle = cycles[int(cycle_index)]
        assigned = distance_cube[int(cycle_index)][int(label_index)]
        nearest_label = np.argmin(
            np.column_stack(distance_cube[int(cycle_index)]), axis=1
        )
        fraction = float(np.mean(nearest_label == int(label_index)))
        record = BoundaryAssignment(
            label=label,
            official_tag=BOUNDARY_TAGS[label],
            raw_boundary_nodes=int(len(cycle)),
            raw_perimeter_mm=_cycle_perimeter(mesh.points, cycle),
            median_distance_mm=float(np.median(assigned)),
            p95_distance_mm=float(np.quantile(assigned, 0.95)),
            maximum_distance_mm=float(np.max(assigned)),
            nearest_tag_fraction=fraction,
        )
        mapping[label] = cycle
        records.append(record)
    if set(mapping) != set(BOUNDARY_TAGS):
        raise ValueError("bilayer-to-raw boundary assignment is incomplete")
    for record in records:
        if (
            record.p95_distance_mm > 0.25
            or record.maximum_distance_mm > 0.30
            or record.nearest_tag_fraction < 0.95
        ):
            raise ValueError(f"unreliable official boundary transfer: {record}")
    return mapping, sorted(records, key=lambda item: item.official_tag)


def _edge_graph(mesh: TriangleMesh) -> csr_matrix:
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    edges = np.sort(
        np.vstack(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
        ),
        axis=1,
    )
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(
        mesh.points[edges[:, 0]] - mesh.points[edges[:, 1]], axis=1
    )
    return coo_matrix(
        (
            np.r_[lengths, lengths],
            (
                np.r_[edges[:, 0], edges[:, 1]],
                np.r_[edges[:, 1], edges[:, 0]],
            ),
        ),
        shape=(mesh.n_vertices, mesh.n_vertices),
    ).tocsr()


def _node_area(mesh: TriangleMesh) -> np.ndarray:
    result = np.zeros(mesh.n_vertices, dtype=float)
    contribution = mesh.triangle_areas() / 3.0
    for local in range(3):
        np.add.at(result, mesh.triangles[:, local], contribution)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("surface has a nonpositive lumped vertex area")
    return result


def _threshold_anchored_score(voltage_mv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    voltage = np.asarray(voltage_mv, dtype=float)
    if voltage.ndim != 1 or not np.isfinite(voltage).all() or np.any(voltage < 0.0):
        raise ValueError("bipolar voltage must be a finite nonnegative vertex field")
    bounded = np.exp2(-voltage / VOLTAGE_THRESHOLD_MV)
    signed = 2.0 * bounded - 1.0
    if np.any(bounded < 0.0) or np.any(bounded > 1.0):
        raise FloatingPointError("threshold-anchored score left [0,1]")
    return np.asarray(bounded), np.asarray(signed)


def _voxel_contacts(
    points: np.ndarray,
    allowed: np.ndarray,
    voxel_size_mm: float,
) -> np.ndarray:
    candidates = np.flatnonzero(allowed)
    if len(candidates) == 0:
        raise ValueError("contact selection has no allowed surface nodes")
    origin = np.min(points, axis=0) - 0.5 * voxel_size_mm
    keys = np.floor((points[candidates] - origin) / voxel_size_mm).astype(np.int64)
    centres = origin + (keys + 0.5) * voxel_size_mm
    distance2 = np.sum((points[candidates] - centres) ** 2, axis=1)
    order = np.lexsort(
        (candidates, distance2, keys[:, 2], keys[:, 1], keys[:, 0])
    )
    ordered_keys = keys[order]
    first = np.r_[True, np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)]
    contacts = candidates[order[first]]
    return np.asarray(np.sort(contacts), dtype=np.int64)


def _observation_fields(
    mesh: TriangleMesh,
    signed_score: np.ndarray,
    blackout: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contacts = _voxel_contacts(mesh.points, ~blackout, CONTACT_VOXEL_SIZE_MM)
    tree = cKDTree(mesh.points)
    confidence = np.zeros(mesh.n_vertices, dtype=float)
    forcing = np.zeros(mesh.n_vertices, dtype=float)
    neighbourhoods = tree.query_ball_point(
        mesh.points[contacts], KERNEL_SUPPORT_MM, workers=1
    )
    for contact, neighbours in zip(contacts, neighbourhoods):
        nodes = np.asarray(neighbours, dtype=np.int64)
        scaled_distance = (
            np.linalg.norm(mesh.points[nodes] - mesh.points[contact], axis=1)
            / KERNEL_SUPPORT_MM
        )
        weight = KERNEL_WEIGHT * (1.0 - scaled_distance) ** 4 * (
            1.0 + 4.0 * scaled_distance
        )
        confidence[nodes] += weight
        forcing[nodes] += weight * signed_score[contact]
    confidence[blackout] = 0.0
    forcing[blackout] = 0.0
    if np.max(np.abs(forcing[blackout])) != 0.0 or np.max(confidence[blackout]) != 0.0:
        raise FloatingPointError("blackout confidence/forcing is not exact zero")
    if np.any(np.abs(forcing) > confidence + 1.0e-12):
        raise FloatingPointError("compact observations violate |forcing| <= confidence")
    return contacts, confidence, forcing


def _prediction_score(state: np.ndarray) -> np.ndarray:
    """Prespecified saturated readout used only for reporting and transfer.

    The phase state itself is never clipped or projected into ``[-1,1]``.
    """
    values = np.asarray(state, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("reconstruction state contains a non-finite value")
    return 0.5 * (1.0 + np.clip(values, -1.0, 1.0))


def _class_boundary_nodes(
    mesh: TriangleMesh,
    evaluation: np.ndarray,
    classification: np.ndarray,
) -> np.ndarray:
    triangles = mesh.triangles
    edges = np.unique(
        np.sort(
            np.vstack(
                (
                    triangles[:, [0, 1]],
                    triangles[:, [1, 2]],
                    triangles[:, [2, 0]],
                )
            ),
            axis=1,
        ),
        axis=0,
    )
    keep = (
        evaluation[edges[:, 0]]
        & evaluation[edges[:, 1]]
        & (classification[edges[:, 0]] != classification[edges[:, 1]])
    )
    return np.unique(edges[keep].ravel()).astype(np.int64)


def _boundary_hd95(
    mesh: TriangleMesh,
    graph: csr_matrix,
    evaluation: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    truth_nodes = _class_boundary_nodes(mesh, evaluation, truth)
    prediction_nodes = _class_boundary_nodes(mesh, evaluation, prediction)
    if len(truth_nodes) == 0 and len(prediction_nodes) == 0:
        return 0.0 if np.array_equal(truth[evaluation], prediction[evaluation]) else np.nan
    if len(truth_nodes) == 0 or len(prediction_nodes) == 0:
        return np.nan
    to_truth = np.asarray(
        dijkstra(graph, directed=False, indices=truth_nodes, min_only=True), dtype=float
    )
    to_prediction = np.asarray(
        dijkstra(graph, directed=False, indices=prediction_nodes, min_only=True),
        dtype=float,
    )
    distances = np.r_[to_truth[prediction_nodes], to_prediction[truth_nodes]]
    return float(np.quantile(distances, 0.95))


def _weighted_calibration(
    reference: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("calibration weights have nonpositive total")
    mean_prediction = float(np.dot(weights, prediction) / total)
    mean_reference = float(np.dot(weights, reference) / total)
    centred = prediction - mean_prediction
    denominator = float(np.dot(weights, centred * centred))
    if denominator <= np.finfo(float).eps * total:
        intercept = np.nan
        slope = np.nan
    else:
        slope = float(np.dot(weights, centred * (reference - mean_reference)) / denominator)
        intercept = mean_reference - slope * mean_prediction
    bins = np.linspace(0.0, 1.0, 11)
    membership = np.minimum(np.searchsorted(bins, prediction, side="right") - 1, 9)
    ece = 0.0
    for index in range(10):
        chosen = membership == index
        weight = float(np.sum(weights[chosen]))
        if weight:
            gap = abs(
                float(np.dot(weights[chosen], prediction[chosen] - reference[chosen]) / weight)
            )
            ece += weight / total * gap
    return float(intercept), float(slope), float(ece)


def _reconstruction_metrics(
    mesh: TriangleMesh,
    graph: csr_matrix,
    node_area: np.ndarray,
    reference_score: np.ndarray,
    reference_voltage: np.ndarray,
    state: np.ndarray,
    evaluation: np.ndarray,
) -> dict[str, float]:
    reference = reference_score[evaluation]
    prediction_full = _prediction_score(state)
    prediction = prediction_full[evaluation]
    weights = node_area[evaluation]
    total = float(np.sum(weights))
    error = prediction - reference
    integrated_squared_error = float(np.dot(weights, error * error))
    mse = integrated_squared_error / total
    mae = float(np.dot(weights, np.abs(error)) / total)
    bias = float(np.dot(weights, error) / total)
    truth_class_full = reference_voltage <= VOLTAGE_THRESHOLD_MV
    prediction_class_full = prediction_full >= 0.5
    truth_class = truth_class_full[evaluation]
    prediction_class = prediction_class_full[evaluation]
    positive = float(np.sum(weights[truth_class]))
    negative = float(np.sum(weights[~truth_class]))
    true_positive = float(np.sum(weights[truth_class & prediction_class]))
    true_negative = float(np.sum(weights[(~truth_class) & (~prediction_class)]))
    predicted_positive = float(np.sum(weights[prediction_class]))
    dice_denominator = positive + predicted_positive
    dice = 1.0 if dice_denominator == 0.0 else 2.0 * true_positive / dice_denominator
    balanced_accuracy = (
        0.5 * (true_positive / positive + true_negative / negative)
        if positive > 0.0 and negative > 0.0
        else np.nan
    )
    intercept, slope, ece = _weighted_calibration(reference, prediction, weights)
    uncensored = np.asarray(state, dtype=float)[evaluation]
    out_of_range = (uncensored < -1.0) | (uncensored > 1.0)
    out_of_range_fraction = float(np.sum(weights[out_of_range]) / total)
    return {
        "evaluation_area_mm2": total,
        "rmse_area_weighted": float(np.sqrt(mse)),
        "mean_squared_error_area_weighted": mse,
        "integrated_squared_error_mm2": integrated_squared_error,
        "mae_area_weighted": mae,
        "score_bias": bias,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "binned_score_calibration_error": ece,
        "uncensored_state_min": float(np.min(uncensored)),
        "uncensored_state_max": float(np.max(uncensored)),
        "uncensored_state_out_of_range_fraction": out_of_range_fraction,
        "dice_voltage_le_0p1": float(dice),
        "balanced_accuracy_voltage_le_0p1": float(balanced_accuracy),
        "boundary_hd95_mm": _boundary_hd95(
            mesh,
            graph,
            evaluation,
            truth_class_full,
            prediction_class_full,
        ),
        "reference_barrier_area_fraction": positive / total,
        "predicted_barrier_area_fraction": predicted_positive / total,
    }


def _cycle_gap_metrics(
    points: np.ndarray,
    cycle: np.ndarray,
    state: np.ndarray,
) -> tuple[float, int]:
    score = _prediction_score(state)[cycle]
    viable = score < 0.5
    closed = np.r_[cycle, cycle[0]]
    edge_lengths = np.linalg.norm(points[closed[1:]] - points[closed[:-1]], axis=1)
    total = float(np.sum(edge_lengths))
    if np.all(viable):
        return total, 1
    if not np.any(viable):
        return 0.0, 0
    # Piecewise-linear threshold crossings give sub-edge arc lengths.  Rotate
    # to begin at a nonviable node so circular components are counted once.
    origin = int(np.flatnonzero(~viable)[0])
    indices = (origin + np.arange(len(cycle))) % len(cycle)
    probability = score[indices]
    lengths = edge_lengths[indices]
    segments: list[float] = []
    current = 0.0
    for index, length in enumerate(lengths):
        left = float(probability[index])
        right = float(probability[(index + 1) % len(probability)])
        if left < 0.5 and right < 0.5:
            current += float(length)
        elif left >= 0.5 and right >= 0.5:
            if current > 0.0:
                segments.append(current)
                current = 0.0
        else:
            crossing = float(np.clip((0.5 - left) / (right - left), 0.0, 1.0))
            viable_fraction = crossing if left < 0.5 else 1.0 - crossing
            current += viable_fraction * float(length)
            if right >= 0.5 and current > 0.0:
                segments.append(current)
                current = 0.0
    if current > 0.0:
        segments.append(current)
    return (float(max(segments)), int(len(segments))) if segments else (0.0, 0)


def _load_patient(
    cohort_root: Path,
    patient_id: str,
) -> tuple[TriangleMesh, dict[str, np.ndarray], list[BoundaryAssignment], dict[str, Any]]:
    patient_dir = cohort_root / patient_id
    vtk_path = patient_dir / f"{patient_id}_with_erp_lat_bi.vtk"
    points, triangles, point_data, cell_data = read_legacy_polydata(vtk_path)
    points, triangles, point_data, cell_data, cleanup = clean_triangular_surface(
        points, triangles, point_data, cell_data
    )
    mesh = TriangleMesh(points, triangles, point_data, cell_data)
    quality = mesh.assert_valid(require_connected=True, require_consistent_orientation=True)
    required = {"bi", "lat", "erp_laplace", "Ids"}
    missing = required.difference(mesh.point_data)
    if missing:
        raise ValueError(f"{patient_id}: raw surface lacks arrays {sorted(missing)}")
    voltage = np.asarray(mesh.point_data["bi"], dtype=float)
    if voltage.shape != (mesh.n_vertices,):
        raise ValueError(f"{patient_id}: point array 'bi' is not scalar")
    boundaries, boundary_records = _assign_official_boundaries(patient_dir, mesh)
    inventory = {
        "patient_id": patient_id,
        "chamber": "LA",
        "prior_pvi": 1,
        "vtk_path": f"data/meshes/{patient_id}/{patient_id}_with_erp_lat_bi.vtk",
        "vtk_sha256": _sha256(vtk_path),
        "bilayer_points_sha256": _sha256(
            patient_dir / "bilayer" / "LA_bilayer_with_fiber_um.pts"
        ),
        "bilayer_elements_sha256": _sha256(
            patient_dir / "bilayer" / "LA_bilayer_with_fiber_um.elem"
        ),
        "n_vertices": mesh.n_vertices,
        "n_triangles": mesh.n_triangles,
        "source_polygons": int(
            np.max(np.asarray(mesh.cell_data["source_polygon_id"], dtype=int)) + 1
        ),
        "source_quadrilateral_children": int(
            np.sum(np.asarray(mesh.cell_data["source_polygon_size"]) == 4)
        ),
        "removed_zero_area_triangles": len(cleanup.removed_zero_area_triangle_indices),
        "removed_duplicate_triangles": len(cleanup.removed_duplicate_triangle_indices),
        "removed_isolated_points": len(cleanup.removed_isolated_point_indices),
        "connected_components": quality.connected_components,
        "boundary_components": quality.boundary_components,
        "nonmanifold_edges": quality.nonmanifold_edges,
        "orientation_conflicts": quality.orientation_conflicts,
        "total_area_mm2": quality.total_area,
        "minimum_angle_degrees": quality.minimum_angle_degrees,
        "maximum_edge_ratio": quality.maximum_edge_ratio,
        "point_arrays": json.dumps(sorted(mesh.point_data)),
        "cell_arrays": json.dumps(sorted(mesh.cell_data)),
        "bi_min_mv": float(np.min(voltage)),
        "bi_median_mv": float(np.median(voltage)),
        "bi_max_mv": float(np.max(voltage)),
        "vertex_fraction_bi_le_0p1": float(np.mean(voltage <= VOLTAGE_THRESHOLD_MV)),
        "boundary_transfer_json": json.dumps(
            [asdict(record) for record in boundary_records], sort_keys=True
        ),
    }
    return mesh, boundaries, boundary_records, inventory


def _phase_parameters(method: str) -> SurfacePhaseParameters:
    common: dict[str, Any] = {
        "mu": PHASE_MU,
        "dt": PHASE_DT,
        "rho_factor": RHO_FACTOR,
        "admm_tolerance": ADMM_TOLERANCE,
        "admm_max_iterations": ADMM_MAX_ITERATIONS,
        "linear_method": "direct",
        "linear_tolerance": LINEAR_TOLERANCE,
        "linear_absolute_tolerance": LINEAR_ABSOLUTE_TOLERANCE,
        "linear_max_iterations": LINEAR_MAX_ITERATIONS,
    }
    if method == "passive":
        return SurfacePhaseParameters.passive(**common)
    if method == "graph":
        return SurfacePhaseParameters.graph(nu=GRAPH_NU, **common)
    raise ValueError(f"unknown phase method {method!r}")


def _run_patient(
    cohort_root_text: str, patient_id: str,
    checkpoint_root_text: str = str(DEFAULT_CHECKPOINT_DIR),
) -> dict[str, Any]:
    # Avoid BLAS oversubscription inside process-level patient parallelism.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    cohort_root = Path(cohort_root_text)
    checkpoint_root = Path(checkpoint_root_text)
    print(f"{patient_id}: loading anatomy and reference capacity", flush=True)
    mesh, boundaries, boundary_records, inventory = _load_patient(
        cohort_root, patient_id
    )
    voltage = np.asarray(mesh.point_data["bi"], dtype=float)
    bounded_reference, signed_reference = _threshold_anchored_score(voltage)
    node_area = _node_area(mesh)
    graph = _edge_graph(mesh)
    scaled_mesh = TriangleMesh(
        mesh.points / REFERENCE_LENGTH_MM,
        mesh.triangles,
    )
    operators = assemble_p1(scaled_mesh, coefficient=1.0)
    capacity_domains: dict[str, Any] = {}
    reference_capacity: dict[str, tuple[float, float]] = {}
    reference_gap: dict[str, tuple[float, int]] = {}
    for pv_label, _ in PV_TAGS.items():
        domain = build_capacity_domain(
            mesh,
            boundaries[pv_label],
            np.ones(mesh.n_vertices, dtype=bool),
            CAPACITY_WIDTH_MM,
        )
        capacity_domains[pv_label] = domain
        reference_capacity[pv_label] = capacity_from_score(
            domain, signed_reference, residual_limit=1.0e-9
        )
        reference_gap[pv_label] = _cycle_gap_metrics(
            mesh.points, boundaries[pv_label], signed_reference
        )

    metric_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    representative: dict[str, np.ndarray] | None = None
    for mask_name, hidden_pvs in MASK_GROUPS.items():
        source_nodes = np.unique(
            np.concatenate([boundaries[label] for label in hidden_pvs])
        ).astype(np.int64)
        distance = np.asarray(
            dijkstra(graph, directed=False, indices=source_nodes, min_only=True),
            dtype=float,
        )
        evaluation = distance <= EVALUATION_WIDTH_MM
        blackout = distance <= BLACKOUT_WIDTH_MM
        guard = blackout & ~evaluation
        contacts, confidence, forcing = _observation_fields(
            mesh, signed_reference, blackout
        )
        def signature_for(method: str, initial: np.ndarray | None) -> dict[str, Any]:
            parameters = (
                {
                    "epsilon": SCREEN_EPSILON, "length_scale": SCREEN_LENGTH_SCALE,
                    "linear_method": "direct", "linear_tolerance": LINEAR_TOLERANCE,
                    "linear_absolute_tolerance": LINEAR_ABSOLUTE_TOLERANCE,
                    "linear_max_iterations": LINEAR_MAX_ITERATIONS,
                } if method == "screened" else asdict(_phase_parameters(method))
            )
            arrays = {
                "raw_mesh_points": mesh.points, "raw_mesh_triangles": mesh.triangles,
                "scaled_mesh_points": scaled_mesh.points,
                "scaled_mesh_triangles": scaled_mesh.triangles,
                "confidence": confidence, "forcing": forcing,
            }
            if initial is not None:
                arrays["initial_state"] = initial
            return input_signature(
                {
                    "patient_id": patient_id, "mask": mask_name, "method": method,
                    "raw_vtk_sha256": inventory["vtk_sha256"],
                    "reference_length_mm": REFERENCE_LENGTH_MM,
                    "parameters": parameters,
                    "steps": 0 if method == "screened" else PHASE_STEPS,
                    "terminal_states": 1 if method == "screened" else TERMINAL_STATES,
                    "initialization": "screened_linear_solve" if initial is None else "screened_state",
                    "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                },
                arrays,
                {name: ROOT / "code" / name for name in (
                    "run_zenodo_pvi_reconstruction.py", "patient_method_checkpoint.py",
                    "surface_phase.py", "surface_fem.py", "surface_mesh.py",
                    "legacy_vtk_polydata.py",
                )},
            )

        states: dict[str, np.ndarray] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for method in ("screened", "passive", "graph"):
            key = f"{patient_id}_{mask_name}_{method}"
            signature = signature_for(method, None if method == "screened" else states["screened"])
            cached = load_completed(checkpoint_root, key, method, signature, mesh.n_vertices)
            if cached is not None:
                states[method], diagnostics[method] = cached
                print(f"{patient_id} {mask_name} {method}: resumed completed method", flush=True)
                continue
            method_start = time.perf_counter()
            print(f"{patient_id} {mask_name} {method}: solving", flush=True)
            if method == "screened":
                screened = screened_reconstruction(
                    operators, confidence, forcing,
                    epsilon=SCREEN_EPSILON, length_scale=SCREEN_LENGTH_SCALE,
                    linear_method="direct", linear_tolerance=LINEAR_TOLERANCE,
                    linear_absolute_tolerance=LINEAR_ABSOLUTE_TOLERANCE,
                    linear_max_iterations=LINEAR_MAX_ITERATIONS,
                )
                states[method] = np.asarray(screened.state, dtype=float)
                diagnostics[method] = {
                    "solver_relative_residual": float(screened.relative_residual),
                    "maximum_state_residual": float(screened.relative_residual),
                    "maximum_admm_iterations": 0.0,
                    "maximum_mass_source_defect": np.nan,
                    "maximum_graph_projection_residual": np.nan,
                }
                save_completed(checkpoint_root, key, method, signature, states[method], diagnostics[method])
                print(f"{patient_id} {mask_name} {method}: completed in {time.perf_counter()-method_start:.1f} s", flush=True)
                continue
            final_state, history = solve_surface_phase(
                operators,
                states["screened"],
                confidence,
                forcing,
                _phase_parameters(method),
                PHASE_STEPS,
                keep_history=True,
            )
            assert history is not None
            terminal = np.mean(
                np.stack(history[-TERMINAL_STATES:], axis=0), axis=0
            )
            states[method] = np.asarray(terminal, dtype=float)
            step_diagnostics = final_state.diagnostics
            diagnostics[method] = {
                "solver_relative_residual": float(
                    max(item["state_relative_residual"] for item in step_diagnostics)
                ),
                "maximum_state_residual": float(
                    max(item["state_relative_residual"] for item in step_diagnostics)
                ),
                "maximum_admm_iterations": float(
                    max(item["admm_iterations"] for item in step_diagnostics)
                ),
                "maximum_mass_source_defect": float(
                    max(item["mass_source_defect"] for item in step_diagnostics)
                ),
                "maximum_graph_projection_residual": float(
                    max(
                        item["graph_projection_relative_residual"]
                        for item in step_diagnostics
                    )
                ),
            }
            save_completed(checkpoint_root, key, method, signature, states[method], diagnostics[method])
            print(f"{patient_id} {mask_name} {method}: completed in {time.perf_counter()-method_start:.1f} s", flush=True)

        for method, state in states.items():
            metrics = _reconstruction_metrics(
                mesh,
                graph,
                node_area,
                bounded_reference,
                voltage,
                state,
                evaluation,
            )
            metric_rows.append(
                {
                    "patient_id": patient_id,
                    "mask": mask_name,
                    "hidden_pvs": "+".join(hidden_pvs),
                    "method": method,
                    "n_contacts": int(len(contacts)),
                    "contact_voxel_size_mm": CONTACT_VOXEL_SIZE_MM,
                    "kernel_support_mm": KERNEL_SUPPORT_MM,
                    "kernel_weight": KERNEL_WEIGHT,
                    "evaluation_width_mm": EVALUATION_WIDTH_MM,
                    "guard_width_mm": GUARD_WIDTH_MM,
                    "blackout_width_mm": BLACKOUT_WIDTH_MM,
                    "evaluation_nodes": int(np.sum(evaluation)),
                    "guard_nodes": int(np.sum(guard)),
                    "zero_confidence_fraction": float(np.mean(confidence == 0.0)),
                    "blackout_confidence_max": float(np.max(confidence[blackout])),
                    "blackout_forcing_max_abs": float(np.max(np.abs(forcing[blackout]))),
                    "observed_area_fraction": float(
                        np.sum(node_area[confidence > 0.0]) / np.sum(node_area)
                    ),
                    "reference_length_mm": REFERENCE_LENGTH_MM,
                    "pseudo_dt": PHASE_DT,
                    "pseudo_horizon": 0.0 if method == "screened" else PHASE_HORIZON,
                    "terminal_states": 1 if method == "screened" else TERMINAL_STATES,
                    **metrics,
                    **diagnostics[method],
                }
            )
            for pv_label in hidden_pvs:
                capacity, residual = capacity_from_score(
                    capacity_domains[pv_label], state, residual_limit=1.0e-9
                )
                reference_value, reference_residual = reference_capacity[pv_label]
                widest, count = _cycle_gap_metrics(
                    mesh.points, boundaries[pv_label], state
                )
                reference_widest, reference_count = reference_gap[pv_label]
                capacity_rows.append(
                    {
                        "patient_id": patient_id,
                        "mask": mask_name,
                        "method": method,
                        "pv_label": pv_label,
                        "official_bilayer_tag": PV_TAGS[pv_label],
                        "annulus_width_mm": CAPACITY_WIDTH_MM,
                        "normalised_capacity": capacity,
                        "reference_normalised_capacity": reference_value,
                        "capacity_absolute_error": abs(capacity - reference_value),
                        "capacity_absolute_log_ratio": abs(
                            float(np.log(capacity / reference_value))
                        ),
                        "capacity_relative_residual": residual,
                        "reference_capacity_relative_residual": reference_residual,
                        "widest_viable_arc_mm": widest,
                        "reference_widest_viable_arc_mm": reference_widest,
                        "widest_viable_arc_absolute_error_mm": abs(
                            widest - reference_widest
                        ),
                        "viable_arc_count": count,
                        "reference_viable_arc_count": reference_count,
                        "viable_arc_count_absolute_error": abs(count - reference_count),
                    }
                )

        if patient_id == "P3" and mask_name == "left_pair":
            representative = {
                "points": mesh.points,
                "triangles": mesh.triangles,
                "bi_mv": voltage,
                "reference_score": bounded_reference,
                "signed_reference": signed_reference,
                "evaluation_mask": evaluation.astype(np.uint8),
                "guard_mask": guard.astype(np.uint8),
                "blackout_mask": blackout.astype(np.uint8),
                "contact_nodes": contacts,
                "confidence": confidence,
                "forcing": forcing,
                "screened_state": states["screened"],
                "passive_state": states["passive"],
                "graph_state": states["graph"],
            }
            for label, cycle in boundaries.items():
                representative[f"boundary_{label}"] = cycle

    return {
        "inventory": inventory,
        "boundary": [
            {"patient_id": patient_id, **asdict(record)} for record in boundary_records
        ],
        "metrics": metric_rows,
        "capacity": capacity_rows,
        "representative": representative,
    }


def _patient_summary(metrics: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "rmse_area_weighted",
        "mean_squared_error_area_weighted",
        "integrated_squared_error_mm2",
        "mae_area_weighted",
        "score_bias",
        "calibration_intercept",
        "calibration_slope",
        "binned_score_calibration_error",
        "uncensored_state_out_of_range_fraction",
        "dice_voltage_le_0p1",
        "balanced_accuracy_voltage_le_0p1",
        "boundary_hd95_mm",
    ]
    summary = (
        metrics.groupby(["patient_id", "method"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    capacity_columns = [
        "capacity_absolute_error",
        "capacity_absolute_log_ratio",
        "widest_viable_arc_absolute_error_mm",
        "viable_arc_count_absolute_error",
    ]
    capacity_summary = (
        capacity.groupby(["patient_id", "method"], sort=True)[capacity_columns]
        .mean()
        .reset_index()
    )
    result = summary.merge(
        capacity_summary, on=["patient_id", "method"], how="inner", validate="one_to_one"
    )
    return result.sort_values(["patient_id", "method"], kind="mergesort").reset_index(
        drop=True
    )


def _exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(differences))))
    permuted = np.abs(np.mean(signs * differences[None, :], axis=1))
    tolerance = 64.0 * np.finfo(float).eps * max(observed, 1.0)
    return float(np.mean(permuted >= observed - tolerance))


def _paired_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    metric_direction = {
        "rmse_area_weighted": "lower",
        "mae_area_weighted": "lower",
        "binned_score_calibration_error": "lower",
        "dice_voltage_le_0p1": "higher",
        "balanced_accuracy_voltage_le_0p1": "higher",
        "capacity_absolute_log_ratio": "lower",
        "widest_viable_arc_absolute_error_mm": "lower",
        "viable_arc_count_absolute_error": "lower",
    }
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for comparator in ("passive", "screened"):
        for metric, preferred in metric_direction.items():
            pivot = summary.pivot(index="patient_id", columns="method", values=metric)
            paired = pivot[["graph", comparator]].dropna()
            differences = (
                paired["graph"].to_numpy(dtype=float)
                - paired[comparator].to_numpy(dtype=float)
            )
            if len(differences) != len(PATIENT_IDS):
                raise ValueError(f"incomplete patient-level pairing for {metric}")
            indices = rng.integers(
                0, len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences))
            )
            bootstrap = np.mean(differences[indices], axis=1)
            lower, upper = np.quantile(bootstrap, (0.025, 0.975))
            graph_better = differences < 0.0 if preferred == "lower" else differences > 0.0
            rows.append(
                {
                    "contrast": f"graph_minus_{comparator}",
                    "metric": metric,
                    "preferred_direction": preferred,
                    "n_patients": len(differences),
                    "graph_mean": float(np.mean(paired["graph"])),
                    "comparator_mean": float(np.mean(paired[comparator])),
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "paired_bootstrap_lower_95": float(lower),
                    "paired_bootstrap_upper_95": float(upper),
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "exact_two_sided_sign_flip_p": _exact_sign_flip_p(differences),
                    "graph_better_patients": int(np.sum(graph_better)),
                    "ties": int(np.sum(differences == 0.0)),
                    "graph_worse_patients": int(np.sum(~graph_better & (differences != 0.0))),
                }
            )
    return pd.DataFrame(rows)


def _project_surface(points: np.ndarray) -> np.ndarray:
    centred = points - np.mean(points, axis=0)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    basis = right[:2].copy()
    for row in range(2):
        pivot = int(np.argmax(np.abs(basis[row])))
        if basis[row, pivot] < 0.0:
            basis[row] *= -1.0
    return centred @ basis.T


def _surface_panel(
    axis: plt.Axes,
    projected: np.ndarray,
    triangles: np.ndarray,
    nodal_values: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    evaluation: np.ndarray | None = None,
    contacts: np.ndarray | None = None,
) -> PolyCollection:
    face_values = np.mean(nodal_values[triangles], axis=1)
    collection = PolyCollection(
        projected[triangles],
        array=face_values,
        cmap=cmap,
        edgecolors="none",
        linewidths=0.0,
    )
    collection.set_clim(vmin, vmax)
    axis.add_collection(collection)
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(title)
    if evaluation is not None:
        boundary_nodes = _class_boundary_nodes(
            TriangleMesh(
                np.column_stack((projected, np.zeros(len(projected)))), triangles
            ),
            np.ones(len(projected), dtype=bool),
            evaluation,
        )
        axis.scatter(
            projected[boundary_nodes, 0],
            projected[boundary_nodes, 1],
            s=0.7,
            c="#cc2f67",
            linewidths=0,
            rasterized=True,
        )
    if contacts is not None:
        axis.scatter(
            projected[contacts, 0],
            projected[contacts, 1],
            s=1.2,
            c="black",
            linewidths=0,
            alpha=0.55,
            rasterized=True,
        )
    return collection


def _make_figure(
    representative_path: Path,
    summary: pd.DataFrame,
    capacity: pd.DataFrame,
    output_base: Path,
) -> None:
    """Render identical data using the anatomical posterior surface camera."""
    try:
        from patient_surface_plot import make_patient_surface_figure
    except ImportError:  # pragma: no cover
        from .patient_surface_plot import make_patient_surface_figure
    make_patient_surface_figure(representative_path, summary, capacity, output_base)


def _provenance(cohort_root: Path, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "experiment": "patient-derived prior-PVI surface reconstruction",
        "dataset": {
            "title": "Atrial Models with Personalized Effective Refractory Period",
            "doi": DATASET_DOI,
            "url": "https://doi.org/10.5281/zenodo.10726677",
            "license": LICENSE,
            "cohort_root_descriptor": "external_data/zenodo_erp/meshes (user-supplied location)",
            "included_patients": list(PATIENT_IDS),
            "excluded_patient": {
                "patient_id": EXCLUDED_RIGHT_ATRIUM,
                "reason": "right-atrium atrial-flutter case; not a prior-PVI LA surface",
            },
        },
        "source_paper": {
            "doi": SOURCE_PAPER_DOI,
            "url": "https://doi.org/10.1093/europace/euae215",
            "published_voltage_semantics": {
                "ablation_lesion_surrogate": "bipolar voltage <0.1 mV",
                "native_fibrosis_surrogate": "0.1-0.5 mV",
                "healthy_tissue_surrogate": ">0.5 mV",
            },
        },
        "target": {
            "name": "threshold-anchored continuous low-voltage barrier score",
            "formula": "b(V)=2^(-V/0.1mV); q=2b-1",
            "interpretation": "bounded score, not a probability",
            "zero_level": "q=0 exactly when V=0.1 mV",
            "state_handling": "phase states are uncensored; (1+clip(u,-1,1))/2 is a prespecified saturated bounded-score reporting/classification readout only; capacity uses the uncensored u in its positive logistic conductivity law",
            "calibration_endpoint": "area-weighted binned score-calibration error; not probability calibration",
            "out_of_range_endpoint": "uncensored_state_out_of_range_fraction: area fraction in each evaluation mask where u lies outside [-1,1]",
        },
        "boundary_transfer": {
            "raw_surface_correspondence": "epicardial bilayer layer",
            "official_tags": BOUNDARY_TAGS,
            "assignment": "minimum-total-median-distance Hungarian assignment",
            "acceptance": {
                "p95_distance_mm_max": 0.25,
                "maximum_distance_mm_max": 0.30,
                "nearest_tag_fraction_min": 0.95,
            },
        },
        "mask_and_observations": {
            "mask_groups": MASK_GROUPS,
            "evaluation_width_mm": EVALUATION_WIDTH_MM,
            "guard_width_mm": GUARD_WIDTH_MM,
            "blackout_width_mm": BLACKOUT_WIDTH_MM,
            "contact_rule": "one closest-to-centre representative vertex per fixed 5-mm 3-D voxel outside blackout; representatives are not asserted to be 5-mm separated",
            "contact_voxel_size_mm": CONTACT_VOXEL_SIZE_MM,
            "kernel": "compact Wendland C2",
            "kernel_support_mm": KERNEL_SUPPORT_MM,
            "kernel_weight": KERNEL_WEIGHT,
            "post_accumulation_blackout_reset": True,
        },
        "numerics": {
            "reference_length_mm": REFERENCE_LENGTH_MM,
            "surface_diffusion_tensor": "isotropic scalar a=1",
            "patient_fibres_used": False,
            "fibre_note": "The simple clinical POLYDATA has no fibre array; bilayer fibres were not transferred into this locked experiment.",
            "screened": {
                "epsilon": SCREEN_EPSILON,
                "length_scale": SCREEN_LENGTH_SCALE,
            },
            "phase": {
                "mu": PHASE_MU,
                "graph_nu": GRAPH_NU,
                "dt": PHASE_DT,
                "steps": PHASE_STEPS,
                "horizon": PHASE_HORIZON,
                "terminal_states": TERMINAL_STATES,
                "rho_factor": RHO_FACTOR,
                "admm_tolerance": ADMM_TOLERANCE,
                "linear_tolerance": LINEAR_TOLERANCE,
            },
            "capacity_width_mm": CAPACITY_WIDTH_MM,
        },
        "inference": {
            "unit": "patient",
            "within_patient_averaging": "two masks; four hidden PVs for capacity/gap metrics",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "exact_test": "all 2^6 paired sign flips",
            "p_value_scope": "secondary, descriptive, and unadjusted for multiplicity",
        },
        "claim_boundary": [
            "The raw clinical voltage map was already interpolated onto each surface; synthetic contact subsampling is not a replay of raw catheter acquisition.",
            "Voltage below 0.1 mV is a lesion surrogate and does not establish bidirectional electrical block.",
            "The cohort has six LA patients and no independent development/holdout split; parameters were transferred without patient-data tuning.",
            "All six LA patients had prior PVI, so this cohort cannot support a recurrence classifier or comparison with untreated controls.",
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": mpl.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": elapsed_seconds,
            "patient_runner_sha256": _sha256(Path(__file__).resolve()),
        },
    }


def _output_schema(
    inventory: pd.DataFrame,
    boundary: pd.DataFrame,
    metrics: pd.DataFrame,
    capacity: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> dict[str, Any]:
    """Return the exact emitted columns plus definitions of derived endpoints."""

    frames = {
        "zenodo_pvi_mesh_inventory.csv": ("patient", inventory),
        "zenodo_pvi_boundary_transfer.csv": ("patient x anatomical boundary", boundary),
        "zenodo_pvi_reconstruction_metrics.csv": ("patient x blackout mask x method", metrics),
        "zenodo_pvi_capacity.csv": ("patient x blackout mask x method x hidden PV", capacity),
        "zenodo_pvi_patient_summary.csv": ("patient x method", summary),
        "zenodo_pvi_patient_contrasts.csv": ("comparator x endpoint", contrasts),
    }
    return {
        "schema_version": "1.0",
        "tables": {
            name: {
                "row_unit": row_unit,
                "columns": list(frame.columns),
                "dtypes": {column: str(frame[column].dtype) for column in frame.columns},
            }
            for name, (row_unit, frame) in frames.items()
        },
        "endpoint_definitions": {
            "reference_score": "b(V)=2^(-V/0.1mV)",
            "reporting_score": "(1+clip(u,-1,1))/2; u itself remains uncensored",
            "rmse_area_weighted": "sqrt(sum_i A_i (bhat_i-b_i)^2 / sum_i A_i) on evaluation vertices",
            "mean_squared_error_area_weighted": "sum_i A_i (bhat_i-b_i)^2 / sum_i A_i on evaluation vertices",
            "integrated_squared_error_mm2": "sum_i A_i (bhat_i-b_i)^2 on evaluation vertices",
            "mae_area_weighted": "sum_i A_i abs(bhat_i-b_i) / sum_i A_i on evaluation vertices",
            "binned_score_calibration_error": "area-weighted sum over ten fixed bhat bins of abs(mean(bhat-b)); this is score, not probability, calibration",
            "uncensored_state_out_of_range_fraction": "sum of vertex area where uncensored u is outside [-1,1], divided by evaluation area",
            "dice_voltage_le_0p1": "area-weighted Dice for reference V<=0.1mV versus bhat>=0.5",
            "balanced_accuracy_voltage_le_0p1": "mean area-weighted sensitivity and specificity at the same threshold",
            "boundary_hd95_mm": "95th percentile symmetric mesh-geodesic distance between class-boundary vertices",
            "normalised_capacity": "P1 surface Dirichlet capacity using eta(u)=1e-3+0.999/(1+exp(8u)), normalized by the viable coefficient-one capacity on the identical annulus",
            "capacity_absolute_log_ratio": "abs(log(reconstructed normalised capacity / full-map normalised capacity))",
            "patient_summary": "arithmetic mean of the two geometry-only masks; capacity/gap fields also average four hidden PVs",
            "patient_contrast": "graph minus comparator after within-patient averaging",
        },
    }


def run(
    cohort_root: Path, output_dir: Path, figure_dir: Path, workers: int,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
) -> None:
    start = time.perf_counter()
    if not cohort_root.is_dir():
        raise FileNotFoundError(f"Zenodo cohort root not found: {cohort_root}")
    # P2 is excluded by the source study's chamber metadata and is deliberately
    # not downloaded by the six-LA subset fetcher. Its directory is not an input.
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    if workers == 1:
        for patient_id in PATIENT_IDS:
            print(f"running {patient_id}", flush=True)
            results[patient_id] = _run_patient(str(cohort_root), patient_id, str(checkpoint_dir))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_patient, str(cohort_root), patient_id, str(checkpoint_dir)): patient_id
                for patient_id in PATIENT_IDS
            }
            for future in as_completed(futures):
                patient_id = futures[future]
                results[patient_id] = future.result()
                print(f"completed {patient_id}", flush=True)
    if set(results) != set(PATIENT_IDS):
        raise RuntimeError("patient execution returned an incomplete cohort")
    inventory_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    representative: dict[str, np.ndarray] | None = None
    for patient_id in PATIENT_IDS:
        result = results[patient_id]
        inventory_rows.append(result["inventory"])
        boundary_rows.extend(result["boundary"])
        metric_rows.extend(result["metrics"])
        capacity_rows.extend(result["capacity"])
        if result["representative"] is not None:
            if representative is not None:
                raise RuntimeError("more than one representative result was returned")
            representative = result["representative"]
    if representative is None:
        raise RuntimeError("representative P3 left-PV result was not returned")

    inventory_path = output_dir / "zenodo_pvi_mesh_inventory.csv"
    boundary_path = output_dir / "zenodo_pvi_boundary_transfer.csv"
    metrics_path = output_dir / "zenodo_pvi_reconstruction_metrics.csv"
    capacity_path = output_dir / "zenodo_pvi_capacity.csv"
    summary_path = output_dir / "zenodo_pvi_patient_summary.csv"
    contrasts_path = output_dir / "zenodo_pvi_patient_contrasts.csv"
    representative_path = output_dir / "zenodo_pvi_representative.npz"
    provenance_path = output_dir / "zenodo_pvi_provenance.json"
    schema_path = output_dir / "zenodo_pvi_output_schema.json"
    _write_csv(inventory_path, inventory_rows)
    _write_csv(boundary_path, boundary_rows)
    _write_csv(metrics_path, metric_rows)
    _write_csv(capacity_path, capacity_rows)
    inventory = pd.DataFrame(inventory_rows)
    boundary = pd.DataFrame(boundary_rows)
    metrics = pd.DataFrame(metric_rows)
    capacity = pd.DataFrame(capacity_rows)
    summary = _patient_summary(metrics, capacity)
    contrasts = _paired_contrasts(summary)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    np.savez_compressed(representative_path, **representative)
    schema_path.write_text(
        json.dumps(
            _output_schema(inventory, boundary, metrics, capacity, summary, contrasts),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - start
    provenance_path.write_text(
        json.dumps(_provenance(cohort_root, elapsed), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _make_figure(
        representative_path,
        summary,
        capacity,
        figure_dir / "fig6_patient_surface",
    )
    print(
        summary.groupby("method")[
            ["rmse_area_weighted", "dice_voltage_le_0p1", "capacity_absolute_log_ratio"]
        ].mean(),
        flush=True,
    )
    print(f"patient-surface experiment completed in {elapsed:.1f} s", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
        help="exact-input completed-method restart files (default: tmp/zenodo_pvi_method_checkpoints)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(len(PATIENT_IDS), max(1, (os.cpu_count() or 2) // 2)),
        help="patient-level worker processes (default: half the detected CPUs, capped at six)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or args.workers > len(PATIENT_IDS):
        raise SystemExit(f"--workers must be between 1 and {len(PATIENT_IDS)}")
    run(args.cohort_root, args.output_dir, args.figure_dir, args.workers, args.checkpoint_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
