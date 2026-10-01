"""Publication figures, cost table and validation for both Experiment 4 extensions."""
from __future__ import annotations
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
import run_reconstruction_extension as r

ROOT=r.ROOT
DATA=ROOT/"data"
FIG=ROOT/"figures"
SOURCES={"support_exclusion":DATA,"boundary_retained":DATA/"reconstruction_boundary_control"}
COLORS={"screened":"#777777","passive":"#d5822d","graph":"#285d8f"}


def save(fig,name):
    for ext in ("pdf","png"):
        temporary=FIG/f"{name}.partial.{ext}"
        fig.savefig(temporary,dpi=400)
        temporary.replace(FIG/f"{name}.{ext}")
    plt.close(fig)


def actual_maps(directory,cases,name):
    with plt.rc_context({"font.size":11,"axes.titlesize":11,"axes.labelsize":11,
                         "xtick.labelsize":10,"ytick.labelsize":10}):
        fig,axes=plt.subplots(2,4,figsize=(7.15,4.35),layout="constrained")
        grid=r.a.reconstruction_grid(241)
        specs={s["geometry"]:s for s in r.a.GEOMETRY_SPECS}
        for i,(geometry,width,angle) in enumerate(cases):
            with np.load(directory/r.filename(geometry,width,angle)) as data:
                hidden=r.a.holdout_mask(grid,width,np.deg2rad(angle))
                for j,method in enumerate(("truth","screened","passive","graph")):
                    field=r.a.geometry_score_field(grid.X,grid.Y,specs[geometry]["gaps"]) if method=="truth" else resample(resample(data[method],241,axis=0),241,axis=1).real
                    ax=axes[i,j]
                    im=ax.pcolormesh(grid.X,grid.Y,field,shading="auto",cmap="coolwarm",vmin=-1,vmax=1,rasterized=True)
                    ax.contour(grid.X,grid.Y,hidden.astype(float),levels=[.5],colors=["#141414"],linewidths=.85,linestyles="--")
                    ax.contour(grid.X,grid.Y,field,levels=[0],colors=["white"],linewidths=.7)
                    if angle==0:ax.set(xlim=(36,54),ylim=(21,39),xticks=[40,50],yticks=[25,35])
                    else:ax.set(xlim=(21,39),ylim=(36,54),xticks=[25,35],yticks=[40,50])
                    ax.set_aspect("equal")
                    if i==0:ax.set_title({"truth":"Reference","screened":"Screened","passive":"Passive","graph":"Graph"}[method],pad=5)
                    if j:ax.tick_params(labelleft=False)
                    if i==1:ax.set_xlabel("x (mm)")
                    if j==0:ax.set_ylabel(("Ring" if geometry=="complete_ring" else ("3-mm gap" if geometry=="narrow_gap" else "9-mm gap"))+f", {width}-mm mask\ny (mm)")
        colorbar=fig.colorbar(im,ax=axes,orientation="horizontal",shrink=.58,aspect=35,pad=.01,ticks=[-1,0,1])
        colorbar.set_label("Score",labelpad=1)
        save(fig,name)


def summary_figure(summaries):
    with plt.rc_context({"font.size":11,"axes.titlesize":11,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10}):
        fig,axes=plt.subplots(2,3,figsize=(7.2,4.9),layout="constrained",sharex=True,sharey="col")
        for i,(regime,s) in enumerate(summaries.items()):
            for method,color in COLORS.items():
                values=s[s.method==method]
                for j,metric in enumerate(["common_core_score_rmse","capacity_abs_error","gap_count_abs_error"]):
                    axes[i,j].plot(values.blackout_arc_width_mm,values[metric],"o-",color=color,label=method,lw=1.4,ms=4)
            for ax in axes[i]:ax.set(xticks=[2,6,12]);ax.grid(alpha=.18)
            axes[i,0].set_ylabel(("Support exclusion" if i==0 else "Boundary retained")+"\nError")
        for ax,title in zip(axes[0],["Fixed-core RMSE","Capacity error","Gap-count error"]):ax.set_title(title)
        for ax in axes[1]:ax.set_xlabel("Missing arc (mm)")
        axes[0,0].legend(frameon=False,loc="lower right")
        save(fig,"fig4_acquisition_size_comparison")


def work_figure(frames):
    markers={2:"o",6:"s",12:"^"}
    with plt.rc_context({"font.size":11,"axes.titlesize":11,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9}):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout="constrained",sharey=True)
        for ax,(regime,frame) in zip(axes,frames.items()):
            p=frame[(frame.reconstruction_n==81)&(frame.pseudo_dt==.01)]
            for (width,method),part in p.groupby(["blackout_arc_width_mm","method"]):
                ax.scatter(part.total_wall_seconds.median(),part.common_core_score_rmse.mean(),s=40,
                           color=COLORS[method],marker=markers[int(width)],label=f"{method}, {width:g} mm")
            ax.set(xscale="log",xlabel="Median elapsed time (s)",title="Support exclusion" if regime=="support_exclusion" else "Boundary retained")
            ax.grid(alpha=.18)
        axes[0].set_ylabel("Mean common-core RMSE")
        from matplotlib.lines import Line2D
        handles=[Line2D([],[],color=v,marker="o",linestyle="none",label=k) for k,v in COLORS.items()]
        handles += [Line2D([],[],color="black",marker=v,linestyle="none",label=f"{k} mm") for k,v in markers.items()]
        fig.legend(handles=handles,loc="outside lower center",ncol=6,frameon=False)
        save(fig,"fig4_reconstruction_work_error")


def validate():
    rows=[]
    for regime,directory in [("support_exclusion",DATA/"reconstruction_extension_fields"),("boundary_retained",DATA/"reconstruction_boundary_fields")]:
        files=list(directory.glob("*.npz")); observed={}
        for path in files:
            with np.load(path) as f:
                meta=json.loads(str(f["metadata_json"]));hidden=f["evaluation_mask"].astype(bool)
                assert np.all(f["confidence"][hidden]==0) and np.all(f["data_forcing"][hidden]==0)
                for method in ("truth","screened","passive","graph"):assert np.isfinite(f[method]).all()
                key=(meta["geometry"],meta["blackout_arc_width_mm"],meta["blackout_angle_deg"])
                obs=(f["contact_x"].copy(),f["contact_y"].copy(),f["contact_score"].copy(),f["contact_indices"].copy())
                if key in observed:
                    for a,b in zip(observed[key],obs):assert np.array_equal(a,b)
                observed[key]=obs
        for spec in r.a.GEOMETRY_SPECS:
            for angle in r.a.GEOMETRY_BLACKOUT_ANGLES_DEG:
                for small,large in [(2,6),(6,12)]:
                    lo=observed[(spec["geometry"],small,angle)];hi=observed[(spec["geometry"],large,angle)]
                    assert set(hi[3])<=set(lo[3])
                    for k in range(3):assert np.array_equal(lo[k][np.isin(lo[3],hi[3])],hi[k])
        rows.append({"regime":regime,"files":len(files),"zero_data_in_all_hidden_regions":True,
                     "same_observations_across_refinements":True,"nested_acquisition_across_widths":True,"finite_fields":True})
    (DATA/"reconstruction_extension_validation.json").write_text(json.dumps(rows,indent=2)+"\n")


def main():
    frames={regime:pd.read_csv(directory/"reconstruction_extension_blocks.csv") for regime,directory in SOURCES.items()}
    summaries={regime:pd.read_csv(directory/"reconstruction_extension_summary.csv") for regime,directory in SOURCES.items()}
    costs=[]
    for regime,frame in frames.items():
        main=frame[(frame.reconstruction_n==81)&(frame.pseudo_dt==.01)]
        for method,p in main.groupby("method"):
            costs.append({"acquisition":regime,"method":method,"n_cases":len(p),
                          **{f"median_{metric}":p[metric].median() for metric in ["solver_cpu_seconds","total_cpu_seconds","total_wall_seconds","mean_admm_iterations","total_admm_iterations"]},
                          "max_state_residual":p.max_state_residual.max(),
                          "max_graph_projection_residual":p.max_graph_projection_residual.max(),
                          "max_mass_defect":p.max_mass_defect.max()})
    pd.DataFrame(costs).to_csv(DATA/"reconstruction_extension_cost.csv",index=False)
    for ext in ("pdf","png"):
        source=FIG/f"fig4_actual_reconstructions.{ext}";target=FIG/f"fig4_actual_reconstructions_full.{ext}"
        if source.exists() and not target.exists():shutil.copy2(source,target)
    actual_maps(DATA/"reconstruction_extension_fields",[("complete_ring",12,0),("wide_gap",12,90)],"fig4_actual_reconstructions")
    actual_maps(DATA/"reconstruction_boundary_fields",[("complete_ring",2,0),("narrow_gap",2,0)],"fig4_boundary_reconstructions")
    summary_figure(summaries);work_figure(frames);validate()
    selected=[]
    for regime,frame in frames.items():
        p=frame[(frame.reconstruction_n==81)&(frame.pseudo_dt==.01)&(frame.geometry=="complete_ring")]
        for (width,method),g in p.groupby(["blackout_arc_width_mm","method"]):
            selected.append({"acquisition":regime,"width_mm":width,"method":method,"n_rotations":len(g),"false_gap_count":int((g.gap_count>0).sum())})
    pd.DataFrame(selected).to_csv(DATA/"reconstruction_complete_ring_failures.csv",index=False)
    print(pd.DataFrame(costs).to_string(index=False))


if __name__=="__main__":main()
