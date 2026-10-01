"""Compare guarded LU reuse and repeated factorization on the same P3 task."""
from __future__ import annotations
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import json
import time
import numpy as np
import run_zenodo_pvi_reconstruction as original
from surface_mesh import TriangleMesh
from surface_fem import assemble_p1
from surface_phase import SurfacePhaseState, surface_phase_step

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = np.load(ROOT/"data"/"zenodo_pvi_representative.npz")
    operators = assemble_p1(TriangleMesh(source["points"], source["triangles"]), coefficient=1.)
    rows = []
    for method in ("passive", "graph"):
        trajectories = {}
        for reuse in (False, True):
            state = SurfacePhaseState(u=source["screened_state"].copy())
            start = time.perf_counter()
            history = []
            for _ in range(120):
                if not reuse:
                    state._linear_solver_cache.clear()
                state = surface_phase_step(operators, state, source["confidence"], source["forcing"],
                                           original._phase_parameters(method))
                history.append(state.u.copy())
            elapsed = time.perf_counter()-start
            trajectories[reuse] = np.asarray(history)
            row = dict(method=method, factor_reuse=reuse, elapsed_seconds=elapsed,
                       steps=120, vertices=operators.n_vertices,
                       linear_factorizations=sum(d["linear_factorizations"] for d in state.diagnostics),
                       solver_reuses=sum(d["linear_solver_reuses"] for d in state.diagnostics),
                       admm_iterations=sum(d["admm_iterations"] for d in state.diagnostics))
            rows.append(row)
            print(row, flush=True)
        same = np.array_equal(trajectories[False], trajectories[True])
        maximum = float(np.max(np.abs(trajectories[False]-trajectories[True])))
        assert same, f"LU reuse changes {method} trajectory by {maximum}"
        for row in rows:
            if row["method"] == method:
                row.update(full_trajectory_bitwise_equal=same, trajectory_max_abs_difference=maximum)
    destination=ROOT/"data"/"patient_extension"/"patient_factor_reuse_benchmark.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({"patient":"P3", "mask":"left_pair", "width_mm":10,
        "timing_scope":"Two sequential runs per method, BLAS threads=1; other experiments may run concurrently. Timings describe this process and are not machine-independent complexity estimates.",
        "results":rows}, indent=2))


if __name__ == "__main__":
    main()
