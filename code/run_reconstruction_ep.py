"""Frozen reconstruction-to-propagation extension; does not alter Experiment 5.

Run with --stage main, ep_refinement, or reconstruction_refinement. Each
forward solve is checkpointed. All fields, including the reference, use the
same score-to-diffusivity map and the original Experiment 5 EP integrator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import numpy as np
import pandas as pd

from run_applied import (
    CENTER, LENGTH, RADIUS, TARGET_SECTOR_HALF_ANGLE, DATA, FIG,
    reconstruction_grid, score_to_fv_diffusivity, ep_readout,
    capacity_metrics, _ring_gap_geometry,
)

CASES = (("complete_ring", 0), ("narrow_gap", 0), ("wide_gap", 90))
WIDTHS = (2, 6, 12)
METHODS = ("reference", "screened", "passive", "graph")
DIRECTIONS = ("exit", "entrance")
FIELD_DIR = DATA / "reconstruction_extension_fields"
OUT = DATA / "reconstruction_ep"
T_END = 210.0
PROTOCOL = {
    "case_selection": "complete_ring at 0deg, narrow_gap at 0deg, wide_gap at 90deg; selected before inspecting EP outcomes",
    "blackout_arc_widths_mm": WIDTHS,
    "methods": METHODS,
    "pacing_directions": DIRECTIONS,
    "production_ep_n": 121,
    "production_ep_dt_ms": 0.02,
    "t_end_ms": T_END,
    "ep_refinement": {"blackout_arc_width_mm": 6, "spatial_n": 161, "temporal_dt_ms": 0.01},
    "reconstruction_refinement": {"blackout_arc_width_mm": 6, "spatial_n": 121, "temporal_dt": 0.005},
    "primary_readouts": ["target activated fraction", "80-percent sector capture", "first arrival conditional on capture"],
    "arrival_error_rule": "Reported only when both reference and reconstruction capture; failures are never assigned an arrival time.",
    "coefficient_transfer": "Common bilinear periodic score interpolation to 3x3 FV subcells, then eta(s), then cell average; harmonic faces.",
    "analysis_unit": "designed geometry, blackout width and pacing direction; descriptive comparisons only",
    "no_parameter_selection": True,
}


def source_path(geometry: str, width: int, angle: int, n: int, dt: float, acquisition: str = "conservative") -> Path:
    directory=FIELD_DIR if acquisition=="conservative" else DATA/"reconstruction_boundary_fields"
    return directory / f"{geometry}_b{width:02d}_a{angle:03d}_n{n:03d}_dt{round(dt*100000):05d}.npz"


def jobs(stage: str) -> list[dict]:
    acquisition="boundary_coverage" if stage.startswith("boundary") else "conservative"
    configurations = [("production", 81, .01, 121, .02)]
    widths = WIDTHS
    if stage in ("ep_refinement","boundary_ep_refinement","boundary_ep_refinement2"):
        configurations = [("ep_spatial", 81, .01, 161, .02), ("ep_temporal", 81, .01, 121, .01)]
        widths = (2,) if stage=="boundary_ep_refinement2" else (6,)
    elif stage == "reconstruction_refinement":
        configurations = [("reconstruction_spatial", 121, .01, 121, .02), ("reconstruction_temporal", 81, .005, 121, .02)]
        widths = (6,)
    result = []
    for study, rn, rdt, en, edt in configurations:
        for geometry, angle in CASES:
            for width in widths:
                for method in METHODS:
                    for direction in DIRECTIONS:
                        result.append(dict(study=("boundary_"+study if acquisition=="boundary_coverage" else study), acquisition=acquisition, geometry=geometry, blackout_arc_width_mm=width,
                                           angle_deg=angle, method=method, direction=direction,
                                           reconstruction_n=rn, reconstruction_dt=rdt, ep_n=en, ep_dt_ms=edt))
    return result


def target_mask(n: int, angle_deg: float, direction: str) -> np.ndarray:
    x = (np.arange(n)+.5)*LENGTH/n
    X,Y=np.meshgrid(x,x,indexing="ij")
    radius=np.hypot(X-CENTER[0],Y-CENTER[1])
    angle=np.arctan2(Y-CENTER[1],X-CENTER[0])-np.deg2rad(angle_deg)
    distance=np.abs(np.angle(np.exp(1j*angle)))
    lo,hi=(20,23) if direction=="exit" else (7,10)
    return (radius>=lo)&(radius<=hi)&(distance<=TARGET_SECTOR_HALF_ANGLE)


def run_one(job: dict) -> dict:
    path=source_path(job["geometry"],job["blackout_arc_width_mm"],job["angle_deg"],job["reconstruction_n"],job["reconstruction_dt"],job.get("acquisition","conservative"))
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as src:
        score=src["truth" if job["method"]=="reference" else job["method"]].copy()
    payload=score.tobytes()+json.dumps({k:job[k] for k in ("angle_deg","direction","ep_n","ep_dt_ms")},sort_keys=True).encode()
    digest=hashlib.sha256(payload).hexdigest()
    checkpoint=OUT / f"solve_{digest[:24]}.npz"
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as saved:
            result=json.loads(str(saved["result"]))
    else:
        grid=reconstruction_grid(score.shape[0])
        start=time.perf_counter()
        diffusivity=score_to_fv_diffusivity(grid,score,job["ep_n"],subcells=3)
        transfer_seconds=time.perf_counter()-start
        start=time.perf_counter()
        result,solution=ep_readout(diffusivity,np.deg2rad(job["angle_deg"]),job["direction"],t_end=T_END,dt=job["ep_dt_ms"])
        result["ep_wall_seconds"]=time.perf_counter()-start
        result["transfer_wall_seconds"]=transfer_seconds
        result["activation_field_sha256"]=hashlib.sha256(solution.activation.tobytes()).hexdigest()
        # Atomic rename permits reference runs shared by blackout widths to
        # finish concurrently without exposing an incomplete checkpoint.
        temporary=checkpoint.with_name(checkpoint.stem+f"_{os.getpid()}.npz")
        np.savez_compressed(temporary,result=json.dumps(result),activation=solution.activation,
                            peak=solution.peak,diffusivity=diffusivity,
                            trace_time=solution.traces["time"],trace_target=solution.traces["target"])
        os.replace(temporary,checkpoint)
    return {**job,**result,"source_file":str(path.relative_to(DATA.parent)),
            "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "field_sha256":hashlib.sha256(score.tobytes()).hexdigest(),
            "checkpoint":str(checkpoint.relative_to(DATA.parent))}


def load_activation(row: pd.Series) -> np.ndarray:
    with np.load(DATA.parent / row["checkpoint"],allow_pickle=False) as saved:
        return saved["activation"].copy()


def summarize() -> None:
    parts=[pd.read_csv(p) for p in sorted(OUT.glob("raw_*.csv"))]
    if not parts:
        return
    frame=pd.concat(parts,ignore_index=True)
    if "acquisition" not in frame:frame["acquisition"]="conservative"
    frame["acquisition"]=frame["acquisition"].fillna("conservative")
    keys=["study","geometry","blackout_arc_width_mm","angle_deg","direction"]
    reference=frame[frame.method=="reference"].set_index(keys)
    comparisons=[]
    for _,row in frame[frame.method!="reference"].iterrows():
        ref=reference.loc[tuple(row[k] for k in keys)]
        both=bool(row.captured_by_horizon and ref.captured_by_horizon)
        ar,ap=load_activation(ref),load_activation(row)
        target=target_mask(int(row.ep_n),row.angle_deg,row.direction)
        joint=target & np.isfinite(ar) & np.isfinite(ap)
        comparisons.append({**row.to_dict(),
            "reference_captured":int(ref.captured_by_horizon),
            "capture_disagreement":int(row.captured_by_horizon!=ref.captured_by_horizon),
            "reference_target_activated_fraction":ref.target_activated_fraction,
            "target_activated_fraction_error":row.target_activated_fraction-ref.target_activated_fraction,
            "absolute_target_activated_fraction_error":abs(row.target_activated_fraction-ref.target_activated_fraction),
            "reference_total_activated_fraction":ref.activated_fraction,
            "total_activated_fraction_error":row.activated_fraction-ref.activated_fraction,
            "reference_crossing_time_ms":ref.crossing_time_ms,
            "both_capture":int(both),
            "conditional_arrival_error_ms":row.crossing_time_ms-ref.crossing_time_ms if both else np.nan,
            "conditional_absolute_arrival_error_ms":abs(row.crossing_time_ms-ref.crossing_time_ms) if both else np.nan,
            "jointly_activated_target_fraction":float(joint.sum()/target.sum()),
            "joint_target_activation_rmse_ms":float(np.sqrt(np.mean((ap[joint]-ar[joint])**2))) if joint.any() else np.nan,
        })
    comp=pd.DataFrame(comparisons)
    frame.to_csv(DATA/"reconstruction_ep_runs.csv",index=False)
    comp.to_csv(DATA/"reconstruction_ep_comparisons.csv",index=False)
    summary=comp.groupby(["study","blackout_arc_width_mm","method"],sort=False).agg(
        n_comparisons=("geometry","size"),capture_disagreements=("capture_disagreement","sum"),
        mean_absolute_target_fraction_error=("absolute_target_activated_fraction_error","mean"),
        n_both_capture=("both_capture","sum"),mean_absolute_conditional_arrival_error_ms=("conditional_absolute_arrival_error_ms","mean"),
        max_absolute_conditional_arrival_error_ms=("conditional_absolute_arrival_error_ms","max"),
        median_ep_wall_seconds=("ep_wall_seconds","median"),
    ).reset_index()
    summary.to_csv(DATA/"reconstruction_ep_summary.csv",index=False)
    production=frame[frame.study.isin(("production","boundary_production"))].copy()
    refinements=[]
    if "acquisition" not in production:production=production.assign(acquisition="conservative")
    frame["acquisition"]=frame["acquisition"].fillna("conservative")
    production["acquisition"]=production["acquisition"].fillna("conservative")
    p_index=production.set_index(["acquisition","geometry","blackout_arc_width_mm","method","direction"])
    for _,row in frame[~frame.study.isin(("production","boundary_production"))].iterrows():
        base=p_index.loc[tuple(row[k] for k in ("acquisition","geometry","blackout_arc_width_mm","method","direction"))]
        refinements.append({**row.to_dict(),"capture_changed_from_production":int(row.captured_by_horizon!=base.captured_by_horizon),
                            "target_fraction_change_from_production":row.target_activated_fraction-base.target_activated_fraction,
                            "arrival_change_from_production_ms":row.crossing_time_ms-base.crossing_time_ms if row.captured_by_horizon and base.captured_by_horizon else np.nan})
    pd.DataFrame(refinements).to_csv(DATA/"reconstruction_ep_refinement.csv",index=False)
    attribution=[]
    for _,row in frame[(frame.acquisition=="boundary_coverage")&(frame.blackout_arc_width_mm==2)].iterrows():
        with np.load(DATA.parent/row.source_file,allow_pickle=False) as src:
            score=src["truth" if row.method=="reference" else row.method].copy()
        topology=_ring_gap_geometry(reconstruction_grid(score.shape[0]),score)
        with np.load(DATA.parent/row.checkpoint,allow_pickle=False) as state:
            coefficient=state["diffusivity"].copy()
        chash=hashlib.sha256(coefficient.tobytes()).hexdigest()
        cpath=OUT/f"capacity_{chash[:24]}.json"
        if cpath.exists():capacity=json.loads(cpath.read_text())
        else:
            capacity,_=capacity_metrics(coefficient)
            temporary=cpath.with_name(cpath.stem+f"_{os.getpid()}.json")
            temporary.write_text(json.dumps(capacity)+"\n");os.replace(temporary,cpath)
        attribution.append({**row.to_dict(),**topology,**capacity})
    pd.DataFrame(attribution).to_csv(DATA/"reconstruction_ep_topology_capacity.csv",index=False)
    print(summary.to_string(index=False),flush=True)
    for acquisition in ("conservative","boundary_coverage"):
        selected=production[production.acquisition==acquisition]
        if len(selected)==72:
            make_figures(selected,comp,acquisition)
            if acquisition=="boundary_coverage":make_figures(selected,comp,acquisition,width=2)


def make_figures(frame: pd.DataFrame, comparisons: pd.DataFrame, acquisition: str = "conservative", width: int = 6) -> None:
    suffix="" if acquisition=="conservative" else "_boundary"
    if width!=6:suffix+=str(width)
    titles={"complete_ring":"Complete ring", "narrow_gap":"3-mm gap", "wide_gap":"9-mm gap"}
    method_titles={"reference":"Reference","screened":"Screened","passive":"Passive","graph":"Direct graph"}
    fig,axes=plt.subplots(3,4,figsize=(11.7,8.2),layout="constrained")
    cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#ececec")
    for i,(geometry,angle) in enumerate(CASES):
        for j,method in enumerate(METHODS):
            ax=axes[i,j]
            row=frame[(frame.geometry==geometry)&(frame.blackout_arc_width_mm==width)&(frame.method==method)&(frame.direction=="exit")].iloc[0]
            activation=load_activation(row)
            im=ax.imshow(np.ma.masked_invalid(activation.T),origin="lower",extent=(0,60,0,60),cmap=cmap,vmin=0,vmax=T_END,interpolation="nearest")
            with np.load(DATA.parent/row.source_file,allow_pickle=False) as src:
                n=src["truth"].shape[0];x=np.arange(n)*60/n
                ax.contour(x,x,src["truth"].T,levels=[0],colors="white",linewidths=.65)
                ax.contour(x,x,src["evaluation_mask"].T,levels=[.5],colors="#e44855",linewidths=1.1)
            ax.add_patch(Circle(CENTER,3.5,fill=False,ec="#f8f8f8",lw=.9,ls="--"))
            ax.add_patch(Wedge(CENTER,23,angle-15,angle+15,width=3,fill=False,ec="#ef9b1b",lw=1.2))
            ax.set_xlim(3,57);ax.set_ylim(3,57);ax.set_aspect("equal")
            ax.set_xticks((10,30,50));ax.set_yticks((10,30,50))
            if i==0:ax.set_title(method_titles[method],fontweight="bold",fontsize=10)
            if j==0:ax.set_ylabel(titles[geometry]+"\ny (mm)")
            if i==2:ax.set_xlabel("x (mm)")
            status="capture" if row.captured_by_horizon else "no capture"
            timing=f"{row.crossing_time_ms:.1f} ms" if row.captured_by_horizon else "—"
            ax.text(.025,.03,f"{status}; {timing}",transform=ax.transAxes,fontsize=7.8,color="black",
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.86,pad=2))
    cb=fig.colorbar(im,ax=axes,shrink=.8,pad=.015);cb.set_label("First activation time (ms); grey: no activation by 210 ms")
    acquisition_title="conservative contact exclusion" if acquisition=="conservative" else "boundary coverage"
    fig.suptitle(f"Reconstruction-to-propagation transfer: {width}-mm blackouts, exit pacing, {acquisition_title}",fontsize=10.8)
    fig.savefig(FIG/f"fig7_reconstruction_ep{suffix}.pdf");fig.savefig(FIG/f"fig7_reconstruction_ep{suffix}.png",dpi=300);plt.close(fig)
    if width!=6:return

    colors={"screened":"#707070","passive":"#347da8","graph":"#b64545"}
    fig,axes=plt.subplots(2,3,figsize=(10.8,5.9),layout="constrained")
    production=comparisons[comparisons.study==("production" if acquisition=="conservative" else "boundary_production")]
    for j,(geometry,_) in enumerate(CASES):
        for method in METHODS[1:]:
            for i,(metric,label) in enumerate((("absolute_target_activated_fraction_error","Absolute target-fraction error"),("conditional_absolute_arrival_error_ms","Absolute arrival error (ms)\nconditional on both capturing"))):
                for direction,marker,ls in (("exit","o","-"),("entrance","s","--")):
                    part=production[(production.geometry==geometry)&(production.method==method)&(production.direction==direction)].sort_values("blackout_arc_width_mm")
                    axes[i,j].plot(part.blackout_arc_width_mm,part[metric],marker=marker,ls=ls,color=colors[method],label=f"{method_titles[method]}, {direction}",ms=4,lw=1.1)
                axes[i,j].set_xticks(WIDTHS);axes[i,j].grid(alpha=.2);axes[i,j].set_xlabel("Blackout arc width (mm)")
                if j==0:axes[i,j].set_ylabel(label)
        axes[0,j].set_title(titles[geometry],fontweight="bold")
        if geometry=="complete_ring":
            axes[1,j].text(.5,.5,"Reference remains non-capturing\nArrival error is undefined",ha="center",va="center",transform=axes[1,j].transAxes,fontsize=9)
    axes[0,2].legend(fontsize=6.7,loc="best")
    fig.savefig(FIG/f"fig7_reconstruction_ep_missingness{suffix}.pdf");fig.savefig(FIG/f"fig7_reconstruction_ep_missingness{suffix}.png",dpi=300);plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("main","ep_refinement","reconstruction_refinement","boundary_main","boundary_ep_refinement","boundary_ep_refinement2","summarize"),default="main")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if not (OUT/"protocol.json").exists():
        (OUT/"protocol.json").write_text(json.dumps({**PROTOCOL,"python":platform.python_version(),"platform":platform.platform()},indent=2)+"\n")
    if args.stage.startswith("boundary") and not (OUT/"boundary_protocol.json").exists():
        (OUT/"boundary_protocol.json").write_text(json.dumps({**PROTOCOL,"acquisition_control":"Same contact pool with exterior contact centres beyond blackout+1mm guard retained; confidence and forcing remain zero in blackout. All other parameters unchanged.","reason":"Separate the conservative contact exclusion margin from missing-sector width; introduced after conservative complete-ring EP failure was observed and before boundary-coverage EP outcomes."},indent=2)+"\n")
    if args.stage=="boundary_ep_refinement2" and not (OUT/"boundary_2mm_control_protocol.json").exists():
        (OUT/"boundary_2mm_control_protocol.json").write_text(json.dumps({"blackout_width_mm":2,"cases":CASES,"methods":METHODS,"directions":DIRECTIONS,"controls":["EP161dt0.02","EP121dt0.01"],"reason":"Added after observing zero-threshold topology recovery with continued EP capture in boundary-covered2mm complete ring, to test whether this qualitative result survives EP spatial and temporal refinement. No parameter selection."},indent=2)+"\n")
    if args.stage!="summarize":
        tasks=jobs(args.stage)
        missing=sorted({str(source_path(j["geometry"],j["blackout_arc_width_mm"],j["angle_deg"],j["reconstruction_n"],j["reconstruction_dt"],j["acquisition"])) for j in tasks if not source_path(j["geometry"],j["blackout_arc_width_mm"],j["angle_deg"],j["reconstruction_n"],j["reconstruction_dt"],j["acquisition"]).exists()})
        if missing:raise FileNotFoundError("Missing reconstruction inputs:\n"+"\n".join(missing))
        rows=[];start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_one,j):j for j in tasks}
            for future in as_completed(futures):
                row=future.result();rows.append(row)
                print(f"{len(rows)}/{len(tasks)} {row['study']} {row['geometry']} b{row['blackout_arc_width_mm']} {row['method']} {row['direction']}: capture={row['captured_by_horizon']} fraction={row['target_activated_fraction']:.4f} arrival={row['crossing_time_ms']:.3f} elapsed={time.perf_counter()-start:.1f}s",flush=True)
                pd.DataFrame(rows).to_csv(OUT/f"partial_{args.stage}.csv",index=False)
        pd.DataFrame(rows).sort_values(["study","geometry","blackout_arc_width_mm","method","direction"]).to_csv(OUT/f"raw_{args.stage}.csv",index=False)
        (OUT/f"partial_{args.stage}.csv").unlink(missing_ok=True)
    summarize()


if __name__=="__main__":main()
