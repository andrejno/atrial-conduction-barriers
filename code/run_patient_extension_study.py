"""Locked missingness/time/mesh extension of Experiment 6.

See PATIENT_EXTENSION_DESIGN_LOCK.md. Existing outputs are never overwritten.
All evaluations after refinement use original nodes, areas and reference values.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import zipfile

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import run_zenodo_pvi_reconstruction as original
from surface_mesh import TriangleMesh
from surface_fem import assemble_p1
from surface_phase import screened_reconstruction, solve_surface_phase
from run_patient_reconstruction import build_capacity_domain, capacity_from_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "patient_extension"
METHODS = ("screened", "passive", "graph")


def save_npz_atomic(path, **arrays):
    temporary = path.with_name(path.name+".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise IOError(f"NPZ checksum failed: {temporary}")
    temporary.replace(path)


def case_key(patient, mask, width, level, dt):
    return f"{patient}_{mask}_w{width:g}_h{level}_dt{dt:g}".replace(".", "p")


def midpoint_subdivision(mesh):
    """Nested P1 refinement on the identical piecewise-planar geometry."""
    tri = mesh.triangles
    all_edges = np.sort(np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
    edges, inverse = np.unique(all_edges, axis=0, return_inverse=True)
    midpoint_ids = mesh.n_vertices + inverse.reshape(3, -1).T
    a, b, c = tri.T
    ab, bc, ca = midpoint_ids.T
    fine_tri = np.concatenate((np.column_stack((a, ab, ca)),
                               np.column_stack((ab, b, bc)),
                               np.column_stack((ca, bc, c)),
                               np.column_stack((ab, bc, ca))))
    points = np.vstack((mesh.points, mesh.points[edges].mean(axis=1)))
    n = mesh.n_vertices
    rows = np.r_[np.arange(n), n + np.repeat(np.arange(len(edges)), 2)]
    cols = np.r_[np.arange(n), edges.ravel()]
    vals = np.r_[np.ones(n), np.full(2 * len(edges), .5)]
    transfer = coo_matrix((vals, (rows, cols)), shape=(len(points), n)).tocsr()
    fine = TriangleMesh(points, fine_tri)
    fine.assert_valid(require_connected=True, require_consistent_orientation=True)
    assert np.allclose(transfer @ mesh.points, fine.points, rtol=0, atol=1e-12)
    return fine, transfer


def _diagnostics(final, elapsed):
    diag = final.diagnostics
    return {
        "elapsed_seconds": elapsed,
        "maximum_state_residual": max(d["state_relative_residual"] for d in diag),
        "maximum_graph_projection_residual": max(d["graph_projection_relative_residual"] for d in diag),
        "maximum_mass_source_defect": max(d["mass_source_defect"] for d in diag),
        "total_admm_iterations": sum(d["admm_iterations"] for d in diag),
        "maximum_admm_iterations": max(d["admm_iterations"] for d in diag),
        "total_linear_iterations": sum(d["linear_iterations"] for d in diag),
        "linear_factorizations": sum(d.get("linear_factorizations", 0) for d in diag),
        "linear_solver_reuses": sum(d.get("linear_solver_reuses", 0) for d in diag),
    }


def run_case(patient, mask, width, level=0, dt=.01, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    key = case_key(patient, mask, width, level, dt)
    json_path = OUT / f"{key}.json"
    if json_path.exists() and zipfile.is_zipfile(OUT/f"{key}.npz") and not force:
        print(key + ": complete checkpoint", flush=True)
        return json.loads(json_path.read_text())
    print(key + ": starting", flush=True)
    mesh, boundaries, _, inventory = original._load_patient(original.DEFAULT_COHORT_ROOT, patient)
    voltage = np.asarray(mesh.point_data["bi"], float)
    bounded, signed = original._threshold_anchored_score(voltage)
    area = original._node_area(mesh)
    graph = original._edge_graph(mesh)
    pvs = original.MASK_GROUPS[mask]
    source = np.unique(np.concatenate([boundaries[pv] for pv in pvs]))
    distance = dijkstra(graph, directed=False, indices=source, min_only=True)
    evaluation = distance <= width
    common = distance <= 3.
    blackout = distance <= width + original.GUARD_WIDTH_MM
    contacts, confidence, forcing = original._observation_fields(mesh, signed, blackout)
    # Explicitly verify unavailable values cannot enter observation construction.
    poisoned = signed.copy()
    poisoned[blackout] = np.nan
    c2, l2, f2 = original._observation_fields(mesh, poisoned, blackout)
    assert np.array_equal(contacts, c2)
    assert np.array_equal(confidence, l2) and np.array_equal(forcing, f2)
    working_mesh = mesh
    solve_conf, solve_force = confidence, forcing
    for _ in range(level):
        working_mesh, transfer = midpoint_subdivision(working_mesh)
        solve_conf = np.asarray(transfer @ solve_conf)
        solve_force = np.asarray(transfer @ solve_force)
    assert np.all(np.abs(solve_force) <= solve_conf + 1e-12)
    t0 = time.perf_counter()
    operators = assemble_p1(working_mesh, coefficient=1.)
    assembly_seconds = time.perf_counter() - t0
    nsteps = round(original.PHASE_HORIZON / dt)
    terminal = round(original.TERMINAL_STATES * original.PHASE_DT / dt)
    assert np.isclose(nsteps * dt, 1.2) and np.isclose(terminal * dt, .12)
    states, diagnostics = {}, {}
    for method in METHODS:
        cp = OUT / f"{key}_{method}.npz"
        jp = OUT / f"{key}_{method}_solver.json"
        if cp.exists() and jp.exists() and not force:
            states[method] = np.load(cp)["state"]
            diagnostics[method] = json.loads(jp.read_text())
            continue
        print(f"{key} {method}: solve {working_mesh.n_vertices} nodes", flush=True)
        start = time.perf_counter()
        if method == "screened":
            result = screened_reconstruction(
                operators, solve_conf, solve_force,
                epsilon=original.SCREEN_EPSILON,
                length_scale=original.SCREEN_LENGTH_SCALE,
                linear_method="direct",
                linear_tolerance=original.LINEAR_TOLERANCE,
                linear_absolute_tolerance=original.LINEAR_ABSOLUTE_TOLERANCE,
                linear_max_iterations=original.LINEAR_MAX_ITERATIONS)
            states[method] = result.state
            diagnostics[method] = {
                "elapsed_seconds": time.perf_counter() - start,
                "maximum_state_residual": float(result.relative_residual),
                "maximum_graph_projection_residual": 0.,
                "maximum_mass_source_defect": 0.,
                "total_admm_iterations": 0, "maximum_admm_iterations": 0,
                "total_linear_iterations": result.iterations,
            }
        else:
            parameters = replace(original._phase_parameters(method), dt=dt)
            final, history = solve_surface_phase(operators, states["screened"], solve_conf, solve_force,
                                                 parameters, nsteps, keep_history=True)
            states[method] = np.mean(history[-terminal:], axis=0)
            diagnostics[method] = _diagnostics(final, time.perf_counter() - start)
        save_npz_atomic(cp, state=states[method])
        jp.write_text(json.dumps(diagnostics[method], indent=2))
        print(f"{key} {method}: {diagnostics[method]['elapsed_seconds']:.1f}s", flush=True)
    metric_rows, capacity_rows = [], []
    for method in METHODS:
        state = states[method][:mesh.n_vertices]
        metadata = dict(patient_id=patient, mask=mask, method=method,
                        evaluation_width_mm=width, mesh_level=level, pseudo_dt=dt,
                        pseudo_horizon=1.2, terminal_duration=.12, terminal_states=terminal,
                        n_vertices=working_mesh.n_vertices, n_triangles=working_mesh.n_triangles,
                        original_n_vertices=mesh.n_vertices, n_contacts=len(contacts),
                        observation_area_fraction=float(area[confidence > 0].sum()/area.sum()),
                        assembly_seconds=assembly_seconds,
                        blackout_confidence_max=float(np.max(confidence[blackout])),
                        blackout_forcing_max_abs=float(np.max(np.abs(forcing[blackout]))),
                        input_hidden_value_poison_check=True, **diagnostics[method])
        for support_name, support in (("native_band", evaluation), ("common_3mm", common)):
            metrics = original._reconstruction_metrics(mesh, graph, area, bounded, voltage, state, support)
            metric_rows.append({**metadata, "metric_support": support_name, **metrics})
        for pv in pvs:
            domain = build_capacity_domain(mesh, boundaries[pv], np.ones(mesh.n_vertices, bool), 10.)
            capacity, residual = capacity_from_score(domain, state, residual_limit=1e-9)
            reference_capacity, reference_residual = capacity_from_score(domain, signed, residual_limit=1e-9)
            widest, count = original._cycle_gap_metrics(mesh.points, boundaries[pv], state)
            ref_widest, ref_count = original._cycle_gap_metrics(mesh.points, boundaries[pv], signed)
            capacity_rows.append({**metadata, "pv_label": pv, "normalised_capacity": capacity,
                "reference_normalised_capacity": reference_capacity,
                "capacity_absolute_error": abs(capacity-reference_capacity),
                "capacity_absolute_log_ratio": abs(float(np.log(capacity/reference_capacity))),
                "capacity_relative_residual": residual,
                "reference_capacity_relative_residual": reference_residual,
                "widest_viable_arc_mm": widest, "reference_widest_viable_arc_mm": ref_widest,
                "widest_viable_arc_absolute_error_mm": abs(widest-ref_widest),
                "viable_arc_count": count, "reference_viable_arc_count": ref_count})
    save_npz_atomic(OUT/f"{key}.npz", points=mesh.points, triangles=mesh.triangles,
                        bi_mv=voltage, reference_score=bounded, signed_reference=signed,
                        evaluation_mask=evaluation, common_evaluation_mask=common,
                        blackout_mask=blackout, guard_mask=blackout & ~evaluation,
                        contact_nodes=contacts, confidence=confidence, forcing=forcing,
                        node_area=area,
                        **{method+"_state": states[method][:mesh.n_vertices] for method in METHODS},
                        **{"boundary_"+pv: nodes for pv, nodes in boundaries.items()})
    report = {"case": key, "metrics": metric_rows, "capacity": capacity_rows,
              "input_vtk_sha256": inventory["vtk_sha256"], "diagnostics": diagnostics}
    json_path.write_text(json.dumps(report, indent=2))
    print(key + ": completed", flush=True)
    return report


def aggregate():
    records = []
    for path in sorted(OUT.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # A concurrent worker may still be committing its record.
        if "metrics" in rec:
            records.append(rec)
    complete_keys = {record["case"] for record in records}
    metrics = pd.DataFrame([row for rec in records for row in rec["metrics"]])
    capacity = pd.DataFrame([row for rec in records for row in rec["capacity"]])
    metrics.to_csv(OUT/"patient_extension_metrics.csv", index=False)
    capacity.to_csv(OUT/"patient_extension_capacity.csv", index=False)
    group = ["patient_id", "method", "evaluation_width_mm", "mesh_level", "pseudo_dt"]
    patient = metrics.groupby(group+["metric_support"], as_index=False).mean(numeric_only=True)
    patient.to_csv(OUT/"patient_extension_patient_summary.csv", index=False)
    patientcap = capacity.groupby(group, as_index=False).mean(numeric_only=True)
    patientcap.to_csv(OUT/"patient_extension_patient_capacity.csv", index=False)
    patient.groupby(["method", "evaluation_width_mm", "mesh_level", "pseudo_dt", "metric_support"]).mean(numeric_only=True).to_csv(OUT/"patient_extension_group_summary.csv")
    patientcap.groupby(["method", "evaluation_width_mm", "mesh_level", "pseudo_dt"]).mean(numeric_only=True).to_csv(OUT/"patient_extension_group_capacity.csv")
    # Numerical sensitivity uses original geometry and identical quadrature.
    numerical = []
    for patient_id in original.PATIENT_IDS:
        for mask in original.MASK_GROUPS:
            base_path = OUT/(case_key(patient_id, mask, 10, 0, .01)+".npz")
            if not base_path.exists():
                continue
            base = np.load(base_path)
            for level, dt in ((0, .005), (1, .01), (1, .005)):
                fine_path = OUT/(case_key(patient_id, mask, 10, level, dt)+".npz")
                if fine_path.stem not in complete_keys:
                    continue
                fine = np.load(fine_path)
                eval_mask = base["evaluation_mask"].astype(bool)
                w = base["node_area"][eval_mask]
                for method in METHODS:
                    difference = original._prediction_score(fine[method+"_state"])[eval_mask] - original._prediction_score(base[method+"_state"])[eval_mask]
                    numerical.append(dict(patient_id=patient_id, mask=mask, method=method,
                        mesh_level=level, pseudo_dt=dt,
                        score_difference_common_node_l2=float(np.sqrt(np.dot(w, difference**2)/w.sum())),
                        score_difference_common_node_linf=float(np.max(np.abs(difference)))))
    pd.DataFrame(numerical).to_csv(OUT/"patient_extension_numerical_state_changes.csv", index=False)
    # Common-node endpoints do not detect every fine-grid mode. Supplement them
    # by a whole-surface state comparison on the nested refined mesh itself.
    fine_changes = []
    coarse_key = case_key("P3", "left_pair", 10, 0, .01)
    coarse_path = OUT/(coarse_key+".npz")
    if coarse_path.exists():
        coarse = np.load(coarse_path)
        fine_mesh, prolong = midpoint_subdivision(TriangleMesh(coarse["points"], coarse["triangles"]))
        fine_weights = original._node_area(fine_mesh)
        for dt in (.01, .005):
            fine_key = case_key("P3", "left_pair", 10, 1, dt)
            for method in METHODS:
                fine_state_path = OUT/(fine_key+f"_{method}.npz")
                if fine_key in complete_keys and fine_state_path.exists():
                    fine_state = np.load(fine_state_path)["state"]
                    delta = fine_state - prolong @ coarse[method+"_state"]
                    fine_changes.append(dict(patient_id="P3", mask="left_pair",method=method,
                        mesh_level=1,pseudo_dt=dt,comparison="fine_state_minus_prolonged_dt0p01_coarse_state",
                        whole_surface_state_l2=float(np.sqrt(np.dot(fine_weights,delta**2)/fine_weights.sum())),
                        whole_surface_state_linf=float(np.max(np.abs(delta)))))
    pd.DataFrame(fine_changes).to_csv(OUT/"patient_extension_fine_mesh_state_changes.csv", index=False)
    provenance = {
        "design_lock_sha256": hashlib.sha256((ROOT/"PATIENT_EXTENSION_DESIGN_LOCK.md").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "cpu_model": next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unavailable") if Path("/proc/cpuinfo").exists() else platform.processor(),
        "blas_threads": 1, "complete_cases": len(records),
        "timing_scope": "Per-method elapsed wall time including linear factorization, excluding mesh assembly and readout; phase time excludes shared screened initialization. Concurrency is recorded in run log.",
        "refinement_scope": "Nested piecewise-planar mesh; original P1 confidence and forcing are prolonged, without new contacts. Common original nodes/areas/reference used for every readout.",
        "runtime_source_sha256": {name: hashlib.sha256((ROOT/"code"/name).read_bytes()).hexdigest() for name in ("surface_phase.py", "surface_fem.py", "surface_mesh.py")},
    }
    (OUT/"patient_extension_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Aggregated {len(records)} complete cases", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Recompute completed cases; use after changing any numerical input or implementation.")
    parser.add_argument("--part", choices=["all", "width", "time", "mesh", "aggregate", "pilot"], default="all")
    args = parser.parse_args()
    jobs = []
    if args.part in ("all", "width"):
        jobs += [(p, mask, width, 0, .01) for width in (3., 5., 10.) for p in original.PATIENT_IDS for mask in original.MASK_GROUPS]
    if args.part in ("all", "time"):
        jobs += [(p, mask, 10., 0, .005) for p in original.PATIENT_IDS for mask in original.MASK_GROUPS]
    if args.part in ("all", "mesh"):
        jobs += [("P3", "left_pair", 10., 1, dt) for dt in (.01, .005)]
    if args.part == "pilot":
        jobs = [("P3", "left_pair", 3., 0, .01)]
    if jobs:
        print(f"Running {len(jobs)} cases, {args.workers} concurrent workers, BLAS threads=1", flush=True)
        if args.workers == 1:
            for job in jobs:
                run_case(*job, force=args.force)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_case, *job, force=args.force) for job in jobs]
                for future in as_completed(futures):
                    future.result()
    aggregate()


if __name__ == "__main__":
    main()
