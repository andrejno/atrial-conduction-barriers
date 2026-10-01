"""Plot all prespecified patient missingness conditions and representative fields."""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patient_surface_plot import anatomical_view, surface_panel, METHOD_COLOURS

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"patient_extension"


def main():
    plt.rcParams.update({"font.size":9,"axes.titlesize":9,"axes.labelsize":9,
                         "pdf.fonttype":42,"ps.fonttype":42})
    records={width:np.load(DATA/f"P3_left_pair_w{width}_h0_dt0p01.npz") for width in (3,10)}
    base=records[3]
    boundaries={key.replace("boundary_",""):base[key] for key in base.files if key.startswith("boundary_")}
    view=anatomical_view(base["points"],base["triangles"],boundaries)
    fig=plt.figure(figsize=(6.5,5.9))
    grid=fig.add_gridspec(4,4,height_ratios=[1,1,.22,1.35],hspace=.18,wspace=.24,
                          left=.08,right=.985,top=.965,bottom=.095)
    letters="abcdefgh"
    mappable=None
    for row,width in enumerate((3,10)):
        rec=records[width]
        for col,method in enumerate(("reference","screened","passive","graph")):
            ax=fig.add_subplot(grid[row,col])
            values=rec["reference_score"] if method=="reference" else .5*(1+np.clip(rec[method+"_state"],-1,1))
            mappable=surface_panel(ax,view,values,
                title=f"({letters[4*row+col]}) {method.capitalize()}",
                evaluation=rec["evaluation_mask"])
            for annotation in ax.texts:
                annotation.set_fontsize(8.2)
            if col==0:
                ax.text(-.14,.5,f"{width}-mm band",transform=ax.transAxes,rotation=90,
                        va="center",ha="center",fontsize=9)
    slot=grid[2,1:3].get_position(fig)
    colorax=fig.add_axes([.40,slot.y0+.8*slot.height,.24,.012])
    cb=fig.colorbar(mappable,cax=colorax,orientation="horizontal",ticks=[0,.5,1])
    cb.set_label("Voltage-derived barrier score",fontsize=8.2,labelpad=1)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=8.2,pad=1)
    metrics=pd.read_csv(DATA/"patient_extension_patient_summary.csv")
    cap=pd.read_csv(DATA/"patient_extension_patient_capacity.csv")
    metrics=metrics[(metrics.mesh_level==0)&np.isclose(metrics.pseudo_dt,.01)&(metrics.metric_support=="common_3mm")]
    cap=cap[(cap.mesh_level==0)&np.isclose(cap.pseudo_dt,.01)]
    ax=fig.add_subplot(grid[3,:2])
    for method,colour in METHOD_COLOURS.items():
        frame=metrics[metrics.method==method]
        for _,person in frame.groupby("patient_id"):
            person=person.sort_values("evaluation_width_mm")
            ax.plot(person.evaluation_width_mm,person.rmse_area_weighted,color=colour,alpha=.2,lw=.7)
        mean=frame.groupby("evaluation_width_mm").rmse_area_weighted.mean()
        ax.plot(mean.index,mean.values,"o-",color=colour,label=method.capitalize(),lw=1.8,ms=4)
    ax.set_title("(i) Common 0–3-mm region",loc="left")
    ax.set_xlabel("Hidden band width (mm)")
    ax.set_ylabel("Score RMSE")
    ax.set_xticks([3,5,10]); ax.grid(alpha=.2)
    ax.legend(frameon=False,fontsize=8.2,ncol=1)
    ax2=fig.add_subplot(grid[3,2:])
    for method,colour in METHOD_COLOURS.items():
        frame=cap[cap.method==method]
        for _,person in frame.groupby("patient_id"):
            person=person.sort_values("evaluation_width_mm")
            ax2.plot(person.evaluation_width_mm,person.capacity_absolute_error,color=colour,alpha=.2,lw=.7)
        mean=frame.groupby("evaluation_width_mm").capacity_absolute_error.mean()
        ax2.plot(mean.index,mean.values,"o-",color=colour,label=method.capitalize(),lw=1.8,ms=4)
    ax2.set_title("(j) Fixed capacity domains",loc="left")
    ax2.set_xlabel("Hidden band width (mm)")
    ax2.set_ylabel("Absolute capacity error")
    ax2.set_xticks([3,5,10]); ax2.grid(alpha=.2)
    for axis in (ax,ax2):
        axis.spines[["top","right"]].set_visible(False)
    left_pos=ax.get_position(); right_pos=ax2.get_position()
    ax.set_position([.12,left_pos.y0,.355,left_pos.height])
    ax2.set_position([.625,right_pos.y0,.35,right_pos.height])
    for suffix in ("pdf","png"):
        fig.savefig(ROOT/"figures"/f"fig6b_patient_extension.{suffix}",dpi=300)
    plt.close(fig)


if __name__=="__main__":
    main()
