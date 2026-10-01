"""Write descriptive paired contrasts and numerical sensitivity tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"patient_extension"


def main():
    m=pd.read_csv(DATA/"patient_extension_patient_summary.csv")
    c=pd.read_csv(DATA/"patient_extension_patient_capacity.csv")
    detailed=pd.read_csv(DATA/"patient_extension_metrics.csv")
    contrasts=[]
    for (width,level,dt,support),frame in m.groupby(["evaluation_width_mm","mesh_level","pseudo_dt","metric_support"]):
        for endpoint in ("rmse_area_weighted","dice_voltage_le_0p1","mae_area_weighted"):
            pivot=frame.pivot(index="patient_id",columns="method",values=endpoint)
            for baseline in ("passive","screened"):
                d=pivot.graph-pivot[baseline]
                contrasts.append(dict(evaluation_width_mm=width,mesh_level=level,pseudo_dt=dt,
                    metric_support=support,endpoint=endpoint,comparison="graph_minus_"+baseline,
                    n_patients=len(d),mean_difference=d.mean(),minimum_difference=d.min(),
                    maximum_difference=d.max(),negative_differences=int((d<0).sum()),
                    positive_differences=int((d>0).sum())))
    pd.DataFrame(contrasts).to_csv(DATA/"patient_extension_descriptive_contrasts.csv",index=False)
    base=m[(m.mesh_level==0)&np.isclose(m.pseudo_dt,.01)&(m.metric_support=="common_3mm")]
    rows=[]
    for width in (3,5,10):
        for method in ("screened","passive","graph"):
            x=base[(base.evaluation_width_mm==width)&(base.method==method)]
            y=c[(c.evaluation_width_mm==width)&(c.method==method)&(c.mesh_level==0)&np.isclose(c.pseudo_dt,.01)]
            rows.append(dict(width_mm=width,method=method,n_patients=len(x),
                common_3mm_rmse=x.rmse_area_weighted.mean(),common_3mm_dice=x.dice_voltage_le_0p1.mean(),
                fixed_domain_capacity_error=y.capacity_absolute_error.mean(),
                mean_capacity=y.normalised_capacity.mean(),mean_reference_capacity=y.reference_normalised_capacity.mean(),
                longest_arc_error_mm=y.widest_viable_arc_absolute_error_mm.mean()))
    table=pd.DataFrame(rows)
    table.to_csv(DATA/"patient_missingness_publication_table.csv",index=False)
    numerical=[]
    for dt in (.01,.005):
        x=m[(m.evaluation_width_mm==10)&(m.mesh_level==0)&np.isclose(m.pseudo_dt,dt)&(m.metric_support=="native_band")]
        means=x.groupby("method").rmse_area_weighted.mean()
        numerical.append(dict(sample="six_patients",mesh_level=0,vertices="7482--16570",pseudo_dt=dt,
                              passive_rmse=means.get("passive",np.nan),graph_rmse=means.get("graph",np.nan),
                              graph_minus_passive=means.get("graph",np.nan)-means.get("passive",np.nan)))
    for level in (0,1):
        for dt in (.01,.005):
            x=detailed[(detailed.patient_id=="P3")&(detailed["mask"]=="left_pair")&
                       (detailed.evaluation_width_mm==10)&(detailed.mesh_level==level)&
                       np.isclose(detailed.pseudo_dt,dt)&(detailed.metric_support=="native_band")]
            means=x.groupby("method").rmse_area_weighted.mean()
            numerical.append(dict(sample="P3_left_pair",mesh_level=level,
                vertices=int(x.n_vertices.iloc[0]) if len(x) else np.nan,pseudo_dt=dt,
                passive_rmse=means.get("passive",np.nan),graph_rmse=means.get("graph",np.nan),
                graph_minus_passive=means.get("graph",np.nan)-means.get("passive",np.nan)))
    numerical=pd.DataFrame(numerical)
    numerical.to_csv(DATA/"patient_numerical_sensitivity_publication_table.csv",index=False)
    # Recomputed original stress test versus supplied outputs (metric equality).
    supplied=pd.read_csv(ROOT/"data"/"zenodo_pvi_reconstruction_metrics.csv")
    computed=detailed[(detailed.mesh_level==0)&np.isclose(detailed.pseudo_dt,.01)&
                      (detailed.evaluation_width_mm==10)&(detailed.metric_support=="native_band")]
    pairs=computed.merge(supplied,on=["patient_id","mask","method"],suffixes=("_new","_original"))
    check={"compared_rows":len(pairs)}
    for endpoint in ("rmse_area_weighted","dice_voltage_le_0p1","mae_area_weighted"):
        check[endpoint+"_max_abs_difference"]=float(np.max(np.abs(pairs[endpoint+"_new"]-pairs[endpoint+"_original"]))) if len(pairs) else None
    (DATA/"original_stress_test_reproduction.json").write_text(json.dumps(check,indent=2))
    print(table.to_string(index=False))
    print(numerical.to_string(index=False))
    print(check)


if __name__=="__main__":
    main()
