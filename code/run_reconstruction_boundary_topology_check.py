"""Resolution check of every boundary-control case with method-discordant topology.

This is an explicitly result-triggered numerical diagnostic, not an efficacy
subset: every production case in which graph and passive gap counts disagree
is checked on both finer grids and with half the pseudo-time step.
"""
import json
import time
import pandas as pd
import run_reconstruction_boundary_control as b

r=b.r
CASES=[(g,6,270) for g in ("complete_ring","narrow_gap","wide_gap","two_gaps")]
SETTINGS=[(121,.01),(161,.01),(81,.005)]
DESIGN={"status":"Result-triggered threshold sensitivity diagnostic", "rule":"All production cases with graph/passive gap-count disagreement", "cases":CASES, "settings":SETTINGS}


def main():
    path=r.a.DATA/"topology_diagnostic_design.json"
    if not path.exists():path.write_text(json.dumps({"design":DESIGN,"frozen_unix_time":time.time()},indent=2)+"\n")
    specs={s["geometry"]:s for s in r.a.GEOMETRY_SPECS}
    rows=[]
    for g,w,a in CASES:
        for n,dt in SETTINGS:rows+=r.run_case(specs[g],w,a,n,dt)
    pd.DataFrame(rows).to_csv(r.a.DATA/"topology_diagnostic_blocks.csv",index=False)
    print(pd.DataFrame(rows)[["geometry","reconstruction_n","pseudo_dt","method","gap_count","truth_gap_count"]].to_string(index=False))


if __name__=="__main__":main()
