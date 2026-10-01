"""Prespecified nested-blackout, discretization and cost extension of Experiment 4.

Original publication artifacts are not overwritten. Run with one BLAS thread:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/run_reconstruction_extension.py
The design JSON is written before any new reconstruction is evaluated. Exact truth
is used for observation generation and evaluation only, never parameter selection.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample
import run_applied as a

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "reconstruction_extension_fields"
DESIGN = {
    "version": 1,
    "blackout_arc_widths_mm": [2, 6, 12],
    "geometries": [s["geometry"] for s in a.GEOMETRY_SPECS],
    "blackout_angles_deg": list(a.GEOMETRY_BLACKOUT_ANGLES_DEG),
    "acquisition_seeds": list(a.GEOMETRY_ACQUISITION_SEEDS),
    "ring_contacts_before_exclusion": 260,
    "background_contacts_before_exclusion": 320,
    "nested_acquisition": "Coordinates and Gaussian noise fixed before exclusion; no refill.",
    "contact_truth": "Analytic geometry field at fixed physical contact coordinates.",
    "truth_role": "Observation generation and final evaluation only; no tuning.",
    "phase_n": 81, "pseudo_dt": 0.01, "horizon": 1.2,
    "terminal_averaging_duration": 0.12,
    "mu": 0.30, "nu_graph": 0.20, "nu_passive": 0.0,
    "guard_mm": a.HOLDOUT_BUFFER, "compact_support_mm": a.KERNEL_SUPPORT,
    "evaluation_grid_n": 241,
    "common_evaluation_core": "Width-2-mm sector including the common 1-mm guard.",
    "refinement_cases": [["complete_ring", 6, 0], ["narrow_gap", 6, 0], ["wide_gap", 6, 90]],
    "spatial_n": [81, 121, 161],
    "temporal_dt_at_n81": [0.01, 0.005],
    "ep_priority_cases": [[g,w,ang] for g,ang in [("complete_ring",0),("narrow_gap",0),("wide_gap",90)] for w in [2,6,12]],
    "aggregation": "Average four rotations within each geometry, then five designed geometries; descriptive, no population inference.",
    "cost": "Process CPU and wall seconds measured per solver; screened initialization included in flow totals; common T is not common cost.",
    "no_outcome_dependent_changes": True,
}


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f8").tobytes()).hexdigest()


def filename(geometry, width, angle, n=81, dt=0.01):
    return f"{geometry}_b{int(width):02d}_a{int(angle):03d}_n{n:03d}_dt{int(round(dt*100000)):05d}.npz"


def contacts(spec, seed, width, angle):
    """Generate first, then exclude: all sizes share a nested acquisition."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(-np.pi, np.pi, 260, endpoint=False)
    radius = a.RADIUS + rng.normal(0, .55, theta.size)
    x = np.r_[a.CENTER[0] + radius*np.cos(theta), rng.uniform(0, a.LENGTH, 320)]
    y = np.r_[a.CENTER[1] + radius*np.sin(theta), rng.uniform(0, a.LENGTH, 320)]
    exact = a.geometry_score_field(x, y, spec["gaps"])
    noise = rng.normal(0, a.CONTACT_NOISE_SD, x.size)
    keep = ~a._points_in_holdout(x, y, width, np.deg2rad(angle))
    return {"train_x": x[keep], "train_y": y[keep], "train_score": (exact+noise)[keep],
            "train_exact": exact[keep], "n_training": int(keep.sum()),
            "n_ring_contacts": int(keep[:260].sum()),
            "noise_rms": float(np.sqrt(np.mean(noise[keep]**2))),
            "coordinate_sha256": digest(np.column_stack((x[keep],y[keep]))),
            "noise_sha256": digest(noise[keep]), "pre_exclusion_coordinate_sha256": digest(np.column_stack((x,y))),
            "contact_indices": np.flatnonzero(keep)}


def timed(call):
    wall, cpu = time.perf_counter(), time.process_time()
    result = call()
    return result, time.perf_counter()-wall, time.process_time()-cpu


def evaluate(spec, width, angle, grid, fields):
    """All score errors use one fixed grid, including every refinement row."""
    ev = a.reconstruction_grid(DESIGN["evaluation_grid_n"])
    truth = a.geometry_score_field(ev.X, ev.Y, spec["gaps"])
    native = a.holdout_mask(ev, width, np.deg2rad(angle))
    core = a.holdout_mask(ev, 2, np.deg2rad(angle))
    ring_truth = a._ring_gap_geometry(ev, truth)
    native_truth = a.geometry_score_field(grid.X, grid.Y, spec["gaps"])
    truth_capacity = a.capacity_metrics(a.score_to_fv_diffusivity(grid, native_truth))[0]["normalized_capacity"]
    rows = []
    for method, field in fields.items():
        interp = resample(resample(field, ev.nx, axis=0), ev.ny, axis=1).real
        ring = a._ring_gap_geometry(ev, interp)
        capacity = a.capacity_metrics(a.score_to_fv_diffusivity(grid, field))[0]
        row = {"method": method, "normalized_capacity": capacity["normalized_capacity"],
               "truth_normalized_capacity": truth_capacity,
               "capacity_abs_error": abs(capacity["normalized_capacity"]-truth_capacity),
               "gap_count": ring["gap_count"], "truth_gap_count": ring_truth["gap_count"],
               "gap_count_abs_error": abs(ring["gap_count"]-ring_truth["gap_count"]),
               "largest_gap_width_mm": ring["largest_gap_width_mm"],
               "truth_largest_gap_width_mm": ring_truth["largest_gap_width_mm"],
               "largest_gap_width_abs_error_mm": abs(ring["largest_gap_width_mm"]-ring_truth["largest_gap_width_mm"]),
               "capacity_relative_residual": capacity["capacity_relative_residual"],
               "capacity_flux_energy_defect": capacity["capacity_flux_energy_defect"],
               "whole_score_rmse": float(np.sqrt(np.mean((interp-truth)**2)))}
        for label, mask in [("masked", native), ("common_core", core)]:
            error = interp[mask]-truth[mask]
            predicted, true = interp[mask]>0, truth[mask]>0
            denominator = int(predicted.sum()+true.sum())
            row.update({f"{label}_score_rmse": float(np.sqrt(np.mean(error**2))),
                        f"{label}_score_mae": float(np.mean(abs(error))),
                        f"{label}_dice": float(2*np.sum(predicted&true)/denominator) if denominator else 1.,
                        f"{label}_n_eval": int(mask.sum())})
        rows.append(row)
    return rows


def run_case(spec, width, angle, n=81, dt=.01, resume=True):
    path = OUT/filename(spec["geometry"],width,angle,n,dt)
    if resume and path.exists():
        with np.load(path) as saved:
            return json.loads(str(saved["rows_json"]))
    grid = a.reconstruction_grid(n)
    seed = a.GEOMETRY_ACQUISITION_SEEDS[list(a.GEOMETRY_BLACKOUT_ANGLES_DEG).index(float(angle))]
    obs = contacts(spec, seed, width, angle)
    confidence, forcing = a.compact_observation_fields(grid, obs, width, np.deg2rad(angle))
    hidden = a.holdout_mask(grid,width,np.deg2rad(angle))
    assert np.max(confidence[hidden]) == 0 and np.max(abs(forcing[hidden])) == 0
    (initial, screen), screen_wall, screen_cpu = timed(lambda:a.screened_initial_state(grid,confidence,forcing))
    fields = {"screened":initial}
    costs = {"screened":{"solver_wall_seconds":screen_wall,"solver_cpu_seconds":screen_cpu,
              "total_wall_seconds":screen_wall,"total_cpu_seconds":screen_cpu,
              "total_admm_iterations":0, "mean_admm_iterations":0.,
              "max_state_residual":screen["screen_relative_residual"]}}
    steps, window = int(round(1.2/dt)), int(round(.12/dt))
    for method in ("passive","graph"):
        (state, history), wall, cpu = timed(lambda:a.run_reconstruction_history(grid,initial,confidence,forcing,method,steps,dt))
        fields[method] = a.terminal_average(history,steps,window)
        iterations = [d["admm_iterations"] for d in state.diagnostics]
        costs[method] = {"solver_wall_seconds":wall,"solver_cpu_seconds":cpu,
                        "total_wall_seconds":screen_wall+wall,"total_cpu_seconds":screen_cpu+cpu,
                        "total_admm_iterations":int(sum(iterations)),
                        "mean_admm_iterations":float(np.mean(iterations)),
                        "max_state_residual":max(d["state_residual"] for d in state.diagnostics),
                        "max_graph_projection_residual":max(d["graph_projection_residual"] for d in state.diagnostics),
                        "max_mass_defect":max(abs(d["mass_defect"]) for d in state.diagnostics)}
    rows = evaluate(spec,width,angle,grid,fields)
    meta = {"geometry":spec["geometry"],"blackout_arc_width_mm":width,"blackout_angle_deg":angle,
            "reconstruction_n":n,"reconstruction_dx_mm":grid.dx,"pseudo_dt":dt,
            "horizon":1.2,"terminal_window_time":.12,"acquisition_seed":seed,
            "n_training":obs["n_training"],"coordinate_sha256":obs["coordinate_sha256"],
            "noise_sha256":obs["noise_sha256"],"holdout_confidence_max":0.,
            "holdout_forcing_max_abs":0.,"evaluation_grid_n":241}
    rows = [{**meta,**row,**costs[row["method"]]} for row in rows]
    np.savez_compressed(path,truth=a.geometry_score_field(grid.X,grid.Y,spec["gaps"]),
        **fields,confidence=confidence,data_forcing=forcing,evaluation_mask=hidden,
        contact_x=obs["train_x"],contact_y=obs["train_y"],contact_score=obs["train_score"],
        contact_indices=obs["contact_indices"],metadata_json=json.dumps(meta),rows_json=json.dumps(rows))
    print(f"written {path.name}: graph CPU {costs['graph']['solver_cpu_seconds']:.2f}s",flush=True)
    return rows


METRICS = ["masked_score_rmse","common_core_score_rmse","masked_dice","common_core_dice",
           "capacity_abs_error","normalized_capacity","truth_normalized_capacity", "gap_count_abs_error",
           "largest_gap_width_abs_error_mm","total_cpu_seconds","total_wall_seconds","mean_admm_iterations"]


def summaries(frame):
    main = frame[(frame.reconstruction_n==81)&(frame.pseudo_dt==.01)]
    units = main.groupby(["blackout_arc_width_mm","geometry","method"],as_index=False)[METRICS].mean()
    units.to_csv(a.DATA/"reconstruction_extension_geometry_units.csv",index=False)
    summary = units.groupby(["blackout_arc_width_mm","method"],as_index=False)[METRICS].mean()
    summary.to_csv(a.DATA/"reconstruction_extension_summary.csv",index=False)
    contrasts=[]
    for width in DESIGN["blackout_arc_widths_mm"]:
        subset = units[units.blackout_arc_width_mm==width]
        for metric in METRICS:
            p=subset.pivot(index="geometry",columns="method",values=metric)
            d=p.graph-p.passive
            contrasts.append({"blackout_arc_width_mm":width,"metric":metric,"graph_minus_passive":d.mean(),
                              "graph_lower_geometries":int((d<0).sum()),"n_geometries":5,
                              "min_paired_difference":d.min(),"max_paired_difference":d.max(),
                              "graph_minus_screened":(p.graph-p.screened).mean()})
    pd.DataFrame(contrasts).to_csv(a.DATA/"reconstruction_extension_contrasts.csv",index=False)
    return main, units, summary


def refinement_summary(frame):
    rows=[]
    for geometry,width,angle in DESIGN["refinement_cases"]:
        sub=frame[(frame.geometry==geometry)&(frame.blackout_arc_width_mm==width)&(frame.blackout_angle_deg==angle)]
        for n,dt in [(81,.01),(121,.01),(161,.01),(81,.005)]:
            part=sub[(sub.reconstruction_n==n)&(sub.pseudo_dt==dt)].set_index("method")
            for metric in ["masked_score_rmse","common_core_score_rmse","capacity_abs_error","gap_count_abs_error"]:
                rows.append({"geometry":geometry,"blackout_arc_width_mm":width,"blackout_angle_deg":angle,
                             "reconstruction_n":n,"pseudo_dt":dt,"metric":metric,
                             "screened":part.at["screened",metric],"passive":part.at["passive",metric],
                             "graph":part.at["graph",metric],"graph_minus_passive":part.at["graph",metric]-part.at["passive",metric]})
    pd.DataFrame(rows).to_csv(a.DATA/"reconstruction_extension_refinement.csv",index=False)
    # Compare fields on the common Fourier evaluation grid, not unequal masks.
    field_rows=[]
    for geometry,width,angle in DESIGN["refinement_cases"]:
        for study,controls,reference in [("spatial",[(81,.01),(121,.01)],(161,.01)),("temporal",[(81,.01)],(81,.005))]:
            with np.load(OUT/filename(geometry,width,angle,*reference)) as target:
                target_fields={m:resample(resample(target[m],241,axis=0),241,axis=1).real for m in ("screened","passive","graph")}
            for n,dt in controls:
                with np.load(OUT/filename(geometry,width,angle,n,dt)) as source:
                    for m in ("screened","passive","graph"):
                        field=resample(resample(source[m],241,axis=0),241,axis=1).real
                        field_rows.append({"geometry":geometry,"study":study,"method":m,"n":n,"pseudo_dt":dt,
                                           "reference_n":reference[0],"reference_dt":reference[1],
                                           "whole_field_rms_difference":float(np.sqrt(np.mean((field-target_fields[m])**2)))})
    pd.DataFrame(field_rows).to_csv(a.DATA/"reconstruction_extension_field_sensitivity.csv",index=False)


def figures(summary):
    colors={"screened":"#777777","passive":"#d5822d","graph":"#285d8f"}
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.0),layout="constrained")
    for method,color in colors.items():
        s=summary[summary.method==method]
        for ax,metric in zip(axes,["common_core_score_rmse","capacity_abs_error","gap_count_abs_error"]):
            ax.plot(s.blackout_arc_width_mm,s[metric],"o-",color=color,label=method,ms=4)
    for ax,title in zip(axes,["(a) Fixed-core score error","(b) Capacity error","(c) Gap-count error"]):
        ax.set(title=title,xlabel="Missing arc (mm)",xticks=[2,6,12]);ax.grid(alpha=.17)
    axes[0].set_ylabel("RMSE");axes[1].set_ylabel("Absolute normalized error");axes[2].set_ylabel("Absolute count error")
    axes[0].legend(frameon=False)
    for ext in ["pdf","png"]:fig.savefig(a.FIG/f"fig4_blackout_size_extension.{ext}",dpi=400)
    plt.close(fig)
    # Actual reconstructions, chosen in the frozen design, with identical scales.
    fig,axes=plt.subplots(4,4,figsize=(10.6,8.4),layout="constrained")
    cases=[("complete_ring",2,0),("complete_ring",12,0),("wide_gap",2,90),("wide_gap",12,90)]
    for i,(g,w,angle) in enumerate(cases):
        with np.load(OUT/filename(g,w,angle)) as data:
            grid=a.reconstruction_grid()
            for j,method in enumerate(["truth","screened","passive","graph"]):
                ax=axes[i,j];im=ax.pcolormesh(grid.X,grid.Y,data[method],shading="auto",cmap="coolwarm",vmin=-1,vmax=1,rasterized=True)
                ax.contour(grid.X,grid.Y,data["evaluation_mask"].astype(float),levels=[.5],colors=["#1e2025"],linewidths=1.0,linestyles="--")
                ax.contour(grid.X,grid.Y,data[method],levels=[0],colors=["#f4f4f4"],linewidths=.7)
                if angle==0:ax.set(xlim=(36,54),ylim=(18,42))
                else:ax.set(xlim=(18,42),ylim=(36,54))
                ax.set_aspect("equal");ax.tick_params(labelsize=6)
                if i==0:ax.set_title({"truth":"Reference","screened":"Screened","passive":"Passive","graph":"Graph"}[method])
                if j==0:ax.set_ylabel(f"{'Complete ring' if g=='complete_ring' else '9-mm gap'}, {w}-mm blackout\ny (mm)",fontsize=7)
                if i==3:ax.set_xlabel("x (mm)")
    fig.colorbar(im,ax=axes,shrink=.62,pad=.01,label="Score; white contour = 0")
    for ext in ["pdf","png"]:fig.savefig(a.FIG/f"fig4_actual_reconstructions.{ext}",dpi=400)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--refresh",action="store_true");parser.add_argument("--figures-only",action="store_true");args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    design_path=a.DATA/"reconstruction_extension_design.json"
    if design_path.exists():
        if json.loads(design_path.read_text())["design"]!=DESIGN:raise RuntimeError("Frozen design mismatch")
    else:
        design_path.write_text(json.dumps({"design":DESIGN,"frozen_unix_time":time.time(),"platform":platform.platform(),"numpy_version":np.__version__},indent=2)+"\n")
    if args.figures_only:
        figures(pd.read_csv(a.DATA/"reconstruction_extension_summary.csv"));return
    specs={s["geometry"]:s for s in a.GEOMETRY_SPECS}
    tasks=[tuple(c) for c in DESIGN["ep_priority_cases"]]
    tasks += [(g,w,int(ang)) for g in specs for w in DESIGN["blackout_arc_widths_mm"] for ang in a.GEOMETRY_BLACKOUT_ANGLES_DEG if (g,w,int(ang)) not in tasks]
    rows=[]
    for g,w,ang in tasks:
        rows+=run_case(specs[g],w,ang,resume=not args.refresh)
    for g,w,ang in DESIGN["refinement_cases"]:
        for n,dt in [(121,.01),(161,.01),(81,.005)]:rows+=run_case(specs[g],w,ang,n,dt,resume=not args.refresh)
    frame=pd.DataFrame(rows)
    frame.to_csv(a.DATA/"reconstruction_extension_blocks.csv",index=False)
    main_frame,units,summary=summaries(frame)
    refinement_summary(frame);figures(summary)
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":main()
