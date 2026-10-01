"""Check that Euclidean contact kernels stay local on the patient surfaces.

Uses the reconstruction experiment's deterministic contacts, masks, cleaned
surfaces and support radius without changing any observations or solutions.
All reported pair statistics exclude vertices in the exact-zero blackout.
Connectivity is checked in the surface-edge graph induced by the complete
Euclidean support ball. Edge-graph distances upper-bound continuous surface
geodesic distances; exceeding the kernel radius alone need not imply a shortcut.

Run from any directory:
    python code/check_patient_contact_geometry.py --cohort-root PATH --out CSV
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

try:
    import run_zenodo_pvi_reconstruction as experiment
except ImportError:  # pragma: no cover
    from . import run_zenodo_pvi_reconstruction as experiment


def check_patient(cohort_root: Path, patient_id: str) -> list[dict]:
    mesh, boundaries, _, _ = experiment._load_patient(cohort_root, patient_id)
    graph = experiment._edge_graph(mesh)
    tree = cKDTree(mesh.points)
    radius = experiment.KERNEL_SUPPORT_MM
    records = []
    for mask_name, labels in experiment.MASK_GROUPS.items():
        source = np.unique(np.concatenate([boundaries[label] for label in labels]))
        distance = dijkstra(graph, directed=False, indices=source, min_only=True)
        blackout = distance <= experiment.BLACKOUT_WIDTH_MM
        contacts = experiment._voxel_contacts(
            mesh.points, ~blackout, experiment.CONTACT_VOXEL_SIZE_MM
        )
        row = {
            "patient_id": patient_id,
            "mask_name": mask_name,
            "kernel_support_mm": radius,
            "n_contacts": len(contacts),
            "n_supported_pairs_before_blackout": 0,
            "n_retained_supported_pairs": 0,
            "n_disconnected_pairs": 0,
            "n_contacts_with_disconnected_pairs": 0,
            "n_pairs_edge_distance_gt_2r": 0,
            "n_contacts_with_edge_distance_gt_2r": 0,
            "maximum_edge_distance_mm": 0.0,
            "n_pairs_edge_distance_gt_r": 0,
            "total_kernel_weight": 0.0,
            "kernel_weight_edge_distance_gt_r": 0.0,
        }
        neighborhoods = tree.query_ball_point(mesh.points[contacts], radius, workers=1)
        for contact, neighborhood in zip(contacts, neighborhoods):
            nodes = np.asarray(neighborhood, dtype=np.int64)
            retained = ~blackout[nodes]
            scaled = np.linalg.norm(mesh.points[nodes] - mesh.points[contact], axis=1) / radius
            weights = experiment.KERNEL_WEIGHT * (1.0 - scaled) ** 4 * (1.0 + 4.0 * scaled)
            _, components = connected_components(
                graph[nodes, :][:, nodes], directed=False, return_labels=True
            )
            contact_component = components[np.flatnonzero(nodes == contact)[0]]
            disconnected = (components != contact_component) & retained
            edge_distance = dijkstra(
                graph, directed=False, indices=int(contact), limit=2.0 * radius
            )[nodes]
            beyond_twice_radius = (~np.isfinite(edge_distance)) & retained
            # Obtain an actual maximum rather than an infinite cutoff marker if
            # a future cohort contains a long geodesic path inside a short ball.
            if np.any(beyond_twice_radius):
                edge_distance = dijkstra(graph, directed=False, indices=int(contact))[nodes]
            beyond_radius = (edge_distance > radius) & retained
            row["n_supported_pairs_before_blackout"] += len(nodes)
            row["n_retained_supported_pairs"] += int(retained.sum())
            row["n_disconnected_pairs"] += int(disconnected.sum())
            row["n_contacts_with_disconnected_pairs"] += int(np.any(disconnected))
            row["n_pairs_edge_distance_gt_2r"] += int(beyond_twice_radius.sum())
            row["n_contacts_with_edge_distance_gt_2r"] += int(np.any(beyond_twice_radius))
            row["maximum_edge_distance_mm"] = max(
                row["maximum_edge_distance_mm"], float(np.max(edge_distance[retained]))
            )
            row["n_pairs_edge_distance_gt_r"] += int(beyond_radius.sum())
            row["total_kernel_weight"] += float(weights[retained].sum())
            row["kernel_weight_edge_distance_gt_r"] += float(weights[beyond_radius].sum())
        row["kernel_weight_fraction_edge_distance_gt_r"] = (
            row["kernel_weight_edge_distance_gt_r"] / row["total_kernel_weight"]
        )
        records.append(row)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, default=experiment.DEFAULT_COHORT_ROOT)
    parser.add_argument("--out", type=Path,
                        default=experiment.DATA_DIR / "zenodo_pvi_contact_geometry.csv")
    args = parser.parse_args()
    records = []
    for patient_id in experiment.PATIENT_IDS:
        records.extend(check_patient(args.cohort_root, patient_id))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} patient-mask rows to {args.out}")
    print(f"Disconnected retained pairs: {sum(r['n_disconnected_pairs'] for r in records)}")
    print(f"Retained pairs beyond twice radius: {sum(r['n_pairs_edge_distance_gt_2r'] for r in records)}")
    print(f"Maximum edge distance: {max(r['maximum_edge_distance_mm'] for r in records):.6f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
