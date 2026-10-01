"""Additional planar-wave resolution check, locked before computation.

The solution is constant across the propagation direction. Two identical cross
cells preserve exactly that invariant subspace of the original finite-volume
scheme. No physical coefficients, stimulus or regression windows change.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from model import EPParameters, solve_monodomain, activation_cv, apd90
from run_core import _stripe_stimulus
ROOT=Path(__file__).resolve().parents[1]
CASES=((.5,.01),(.25,.01),(.125,.01),(.125,.005))
def main():
    design={'space_time_cases':CASES,'directions':['x','y'],'propagation_length_mm':70,'cross_cells':2,'horizon_ms':350,'fit_interval_mm':[10,55],'stimulus_duration_ms':2,'stimulus_width_mm':2,'stimulus_amplitude':1.2,'purpose':'Planar invariant-subspace spatial sensitivity with finest-grid temporal control; no fitted coefficients.'}
    (ROOT/'data/conduction_extension_design.json').write_text(json.dumps(design,indent=2)+'\n')
    rows=[]
    old=pd.read_csv(ROOT/'data/conduction_calibration.csv')
    for dx,dt in CASES:
        for axis in ('x','y'):
            n=round(70/dx);shape=(n,2) if axis=='x' else (2,n)
            probe=(int(50/dx),1) if axis=='x' else (1,int(30/dx))
            p=EPParameters(dx=dx,dy=dx,dt=dt,d_long=.32,d_trans=.0512)
            start=time.perf_counter()
            sol=solve_monodomain(-np.ones(shape),p,350,_stripe_stimulus(shape,axis),trace_points={'probe':probe})
            elapsed=time.perf_counter()-start
            cv,r2,nfit=activation_cv(sol.activation,dx,0 if axis=='x' else 1,(10.,55.))
            row={'dx_mm':dx,'dt_ms':dt,'direction':axis,'cv_m_per_s':cv,'APD90_ms':apd90(sol.traces['time'],sol.traces['probe']),'regression_R2':r2,'n_fit':nfit,'wall_seconds':elapsed,**sol.diagnostics}
            match=old[(old.study=='space') & (old.dx_mm==dx) & (old.dt_ms==dt) & (old.direction==axis)]
            if len(match):
                row['cv_full_sheet_abs_difference']=abs(cv-match.iloc[0].cv_m_per_s)
                row['APD90_full_sheet_abs_difference']=abs(row['APD90_ms']-match.iloc[0].APD90_ms)
                assert row['cv_full_sheet_abs_difference']<1e-10
                assert row['APD90_full_sheet_abs_difference']<1e-8
            rows.append(row)
            pd.DataFrame(rows).to_csv(ROOT/'data/conduction_extension.csv',index=False)
            print(axis,dx,dt,cv,row['APD90_ms'],elapsed,flush=True)
if __name__=='__main__':main()
