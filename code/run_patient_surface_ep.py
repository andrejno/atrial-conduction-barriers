"""Locked P3 left-PV reconstruction-to-EP extension and solver verification.

Run verification first, then --fields width:path ... .  Every input field is
passed without clipping through the fixed logistic diffusivity law. Geometry,
pacing, kinetics, target region and horizon are fixed across all methods.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import coo_matrix
from surface_mesh import TriangleMesh,rectangular_tri_mesh
from surface_fem import assemble_p1
from surface_ep import solve_surface_ep
from model import EPParameters,_gate_coefficients

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'


def verify():
    rows=[]
    # Exact zero-flux heat solution verifies metric assembly, mass lumping,
    # backward diffusion, forcing-free conservation and spatial consistency.
    D=.128; T=.5
    for n in (8,16,32,64):
        mesh=rectangular_tri_mesh(n,n,length=1,width=1)
        mode=np.cos(np.pi*mesh.points[:,0])*np.cos(np.pi*mesh.points[:,1])
        initial=.4+.1*mode; exact=.4+.1*np.exp(-2*D*np.pi**2*T)*mode
        solution=solve_surface_ep(mesh,np.zeros(mesh.n_vertices),dt=.000025,t_end=T,
                                 v0=initial,diffusion_only=True,constant_diffusivity=True,
                                 snapshot_times=())
        area=assemble_p1(mesh).vertex_area
        rows.append({'study':'heat_spatial','n':n,'dt_ms':.000025,
                     'weighted_l2_error':float(np.sqrt(area@((solution.v-exact)**2)/area.sum())),
                     'mass_drift':float(abs(area@(solution.v-initial))/area.sum()),
                     **solution.diagnostics})
    # Spatially constant full kinetics agree with a high-accuracy independent
    # adaptive ODE solve.  This isolates reaction/gate splitting from diffusion.
    p=EPParameters()
    def rhs(t,y):
        a,b=_gate_coefficients(np.array([y[0]]),p)
        return [y[1]*y[0]**2*(1-y[0])/p.tau_in-y[0]/p.tau_out,a[0]-b[0]*y[1]]
    exact=solve_ivp(rhs,(0,5),[.2,1],method='DOP853',rtol=1e-12,atol=1e-14).y[:,-1]
    mesh=rectangular_tri_mesh(4,4,length=1,width=1)
    for dt in (.1,.05,.025,.0125):
        sol=solve_surface_ep(mesh,np.zeros(mesh.n_vertices),dt=dt,t_end=5,
                             v0=np.full(mesh.n_vertices,.2),diffusion_only=False,
                             constant_diffusivity=True,snapshot_times=())
        error=float(np.linalg.norm([sol.v.mean()-exact[0],sol.h.mean()-exact[1]]))
        rows.append({'study':'kinetics_temporal','n':4,'dt_ms':dt,
                     'weighted_l2_error':error,'constant_spread':float(np.ptp(sol.v)),**sol.diagnostics})
    frame=pd.DataFrame(rows)
    for study in frame.study.unique():
        ids=frame.index[frame.study==study]
        frame.loc[ids,'observed_order']=np.r_[np.nan,np.log2(frame.loc[ids,'weighted_l2_error'].to_numpy()[:-1]/frame.loc[ids,'weighted_l2_error'].to_numpy()[1:])]
    frame.to_csv(DATA/'patient_surface_ep_verification.csv',index=False)
    print(frame[['study','n','dt_ms','weighted_l2_error','observed_order']].to_string(index=False),flush=True)
    if not (frame[frame.study=='heat_spatial'].weighted_l2_error.iloc[-1]<1e-4 and
            frame[frame.study=='kinetics_temporal'].observed_order.iloc[-1]>.9):
        raise RuntimeError('surface EP verification gate failed')


def geometry(data):
    mesh=TriangleMesh(data['points'],data['triangles'])
    edges=np.vstack((mesh.triangles[:,[0,1]],mesh.triangles[:,[1,2]],mesh.triangles[:,[2,0]]))
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    lengths=np.linalg.norm(mesh.points[edges[:,0]]-mesh.points[edges[:,1]],axis=1)
    G=coo_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(mesh.n_vertices,)*2).tocsr()
    distance=dijkstra(G,directed=False,indices=data['boundary_LSPV'],min_only=True)
    stimulus=distance<=2.; target=distance>=20.
    if not stimulus.any() or not target.any(): raise ValueError('empty anatomical protocol region')
    return mesh,distance,stimulus,target


def run_fields(items,dt,halved,overwrite):
    rows=[]
    for item in items:
        width_text,path_text=item.split(':',1); width=float(width_text); path=Path(path_text)
        with np.load(path) as z: data={k:z[k] for k in z.files}
        mesh,distance,stimulus,target=geometry(data); area=assemble_p1(mesh).vertex_area
        fields={'reference':2*data['reference_score']-1}
        fields.update({k:data[k+'_state'] for k in ('screened','passive','graph')})
        np.savez_compressed(DATA/f'patient_surface_ep_geometry_w{width:g}.npz',**data,
                            distance_LSPV_mm=distance,stimulus_mask=stimulus,target_mask=target,vertex_area=area)
        outputs={}
        timesteps=(dt,dt/2) if halved else (dt,)
        for step in timesteps:
            for method,score in fields.items():
                output=DATA/f'patient_surface_ep_w{width:g}_{method}_dt{step:g}.npz'
                meta=output.with_suffix('.json')
                if output.exists() and meta.exists() and not overwrite:
                    z=np.load(output); activation=z['activation']; diag=json.loads(meta.read_text())
                else:
                    print(f'P3 width {width:g} {method} dt {step:g}: start',flush=True)
                    sol=solve_surface_ep(mesh,score,dt=step,stimulus_mask=stimulus)
                    activation=sol.activation; diag=sol.diagnostics
                    np.savez_compressed(output,v=sol.v,h=sol.h,activation=activation,
                                        **{f'v_t{t:g}':v for t,v in sol.snapshots.items()})
                    meta.write_text(json.dumps(diag,indent=2))
                    print(f'  done in {diag["wall_seconds"]:.1f}s, activated {diag["activation_area_fraction"]:.4f}, voltage [{diag["v_min"]:.4g},{diag["v_max"]:.4g}]',flush=True)
                outputs[(step,method)]=activation
                captured=target&np.isfinite(activation)
                target_fraction=float(area[captured].sum()/area[target].sum())
                mean=float(area[captured]@activation[captured]/area[captured].sum()) if captured.any() else np.nan
                rows.append({'patient_id':'P3','mask':'left_pair','width_mm':width,'method':method,
                             'target_activation_area_fraction':target_fraction,
                             'target_mean_arrival_ms':mean,
                             'target_capture_90_percent':int(target_fraction>=.9),**diag})
        for row in rows:
            if row['width_mm']!=width: continue
            a=outputs[(row['dt_ms'],row['method'])]; ref=outputs[(row['dt_ms'],'reference')]
            common=target&np.isfinite(a)&np.isfinite(ref)
            row['target_activation_disagreement_area_fraction']=float(area[target&(np.isfinite(a)!=np.isfinite(ref))].sum()/area[target].sum())
            row['target_common_activation_area_fraction']=float(area[common].sum()/area[target].sum())
            row['target_arrival_rmse_ms']=float(np.sqrt(area[common]@((a[common]-ref[common])**2)/area[common].sum())) if common.any() else np.nan
            row['target_arrival_bias_ms']=float(area[common]@(a[common]-ref[common])/area[common].sum()) if common.any() else np.nan
            allcommon=target.copy()
            for name in fields:
                allcommon &= np.isfinite(outputs[(row['dt_ms'],name)])
            row['target_all_method_common_area_fraction']=float(area[allcommon].sum()/area[target].sum())
            row['target_arrival_common_support_rmse_ms']=float(np.sqrt(area[allcommon]@((a[allcommon]-ref[allcommon])**2)/area[allcommon].sum())) if allcommon.any() else np.nan
            row['target_arrival_common_support_bias_ms']=float(area[allcommon]@(a[allcommon]-ref[allcommon])/area[allcommon].sum()) if allcommon.any() else np.nan
            if halved:
                finer=outputs[(dt/2,row['method'])]; commonfine=target&np.isfinite(a)&np.isfinite(finer)
                row['target_arrival_dt_difference_rmse_ms']=float(np.sqrt(area[commonfine]@((a[commonfine]-finer[commonfine])**2)/area[commonfine].sum())) if commonfine.any() else np.nan
                row['target_activation_dt_disagreement_fraction']=float(area[target&(np.isfinite(a)!=np.isfinite(finer))].sum()/area[target].sum())
    frame=pd.DataFrame(rows); frame.to_csv(DATA/'patient_surface_ep_metrics.csv',index=False)
    print(frame[['width_mm','method','dt_ms','target_activation_area_fraction','target_mean_arrival_ms','target_arrival_rmse_ms','target_arrival_dt_difference_rmse_ms']].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',action='store_true');p.add_argument('--fields',nargs='*',default=[])
    p.add_argument('--dt',type=float,default=.025);p.add_argument('--halved-dt',action='store_true')
    p.add_argument('--overwrite',action='store_true');p.add_argument('--spatial-control',action='store_true');p.add_argument('--plot',action='store_true');a=p.parse_args()
    if a.verify: verify()
    if a.fields: run_fields(a.fields,a.dt,a.halved_dt,a.overwrite)
    if a.spatial_control: spatial_control(a.dt)
    if a.plot: plot_results()



def spatial_control(dt=.025):
    from run_patient_extension_study import midpoint_subdivision
    from model import diffusivity_factor
    rows=[]
    for width in (3.,10.):
        data=dict(np.load(DATA/f'patient_surface_ep_geometry_w{width:g}.npz'))
        mesh=TriangleMesh(data['points'],data['triangles']); fine,transfer=midpoint_subdivision(mesh)
        area=data['vertex_area']; target=data['target_mask'].astype(bool)
        for method in ('reference','graph'):
            signed=2*data['reference_score']-1 if method=='reference' else data['graph_state']
            coefficient=.128*diffusivity_factor(signed)
            path=DATA/f'patient_surface_ep_w{width:g}_{method}_h1_dt{dt:g}.npz'
            meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                a=np.load(path)['activation']; diag=json.loads(meta.read_text())
            else:
                print(f'spatial width{width:g} {method}: {fine.n_vertices} vertices',flush=True)
                sol=solve_surface_ep(fine,transfer@signed,dt=dt,nodal_diffusivity=transfer@coefficient,
                                     stimulus_mask=transfer@data['stimulus_mask'].astype(float))
                a=sol.activation; diag=sol.diagnostics
                np.savez_compressed(path,activation=a,v=sol.v,h=sol.h)
                meta.write_text(json.dumps(diag,indent=2))
                print(f'  spatial done in{diag["wall_seconds"]:.1f}s',flush=True)
            original=np.load(DATA/f'patient_surface_ep_w{width:g}_{method}_dt{dt:g}.npz')['activation']
            a=a[:mesh.n_vertices]; common=target&np.isfinite(a)&np.isfinite(original)
            activated=target&np.isfinite(a)
            rows.append({'width_mm':width,'method':method,'readout_quadrature':'original vertices and lumped areas',
                         'target_activation_area_fraction':float(area[activated].sum()/area[target].sum()),
                         'target_mean_arrival_ms':float(area[activated]@a[activated]/area[activated].sum()),
                         'target_arrival_mesh_difference_rmse_ms':float(np.sqrt(area[common]@((a[common]-original[common])**2)/area[common].sum())),
                         'target_activation_mesh_disagreement_fraction':float(area[target&(np.isfinite(a)!=np.isfinite(original))].sum()/area[target].sum()),**diag})
        ref=np.load(DATA/f'patient_surface_ep_w{width:g}_reference_h1_dt{dt:g}.npz')['activation'][:mesh.n_vertices]
        gr=np.load(DATA/f'patient_surface_ep_w{width:g}_graph_h1_dt{dt:g}.npz')['activation'][:mesh.n_vertices]
        common=target&np.isfinite(ref)&np.isfinite(gr)
        coarse_ref=np.load(DATA/f'patient_surface_ep_w{width:g}_reference_dt{dt:g}.npz')['activation']
        coarse_gr=np.load(DATA/f'patient_surface_ep_w{width:g}_graph_dt{dt:g}.npz')['activation']
        allcommon=common&np.isfinite(coarse_ref)&np.isfinite(coarse_gr)
        for r in rows:
            if r['width_mm']==width:
                r['target_common_native_refined_area_fraction']=float(area[allcommon].sum()/area[target].sum())
                r['native_graph_reference_rmse_common_native_refined_ms']=float(np.sqrt(area[allcommon]@((coarse_gr[allcommon]-coarse_ref[allcommon])**2)/area[allcommon].sum()))
                r['refined_graph_reference_rmse_common_native_refined_ms']=float(np.sqrt(area[allcommon]@((gr[allcommon]-ref[allcommon])**2)/area[allcommon].sum()))
                r['target_arrival_rmse_vs_refined_reference_ms']=float(np.sqrt(area[common]@((gr[common]-ref[common])**2)/area[common].sum())) if r['method']=='graph' else 0.
    pd.DataFrame(rows).to_csv(DATA/'patient_surface_ep_mesh_control.csv',index=False)
    print(pd.DataFrame(rows)[['width_mm','method','target_activation_area_fraction','target_arrival_mesh_difference_rmse_ms','target_arrival_rmse_vs_refined_reference_ms']].to_string(index=False),flush=True)


def plot_results():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection,LineCollection
    from patient_surface_plot import anatomical_view,_contour_segments,_style
    _style()
    frame=pd.read_csv(DATA/'patient_surface_ep_metrics.csv');dt=float(frame.dt_ms.min())
    figure=plt.figure(figsize=(4.8,4.45),layout='constrained')
    grid=figure.add_gridspec(3,2,height_ratios=(1.,1.,.065),hspace=.03,wspace=.02)
    norm=mpl.colors.Normalize(0,210);cmap=mpl.colormaps['turbo']
    for row,width in enumerate((3.,10.)):
        z=np.load(DATA/f'patient_surface_ep_geometry_w{width:g}.npz')
        bounds={label:z['boundary_'+label] for label in ('LSPV','LIPV','RSPV','RIPV')}
        view=anatomical_view(z['points'],z['triangles'],bounds)
        for column,method in enumerate(('reference','graph')):
            ax=figure.add_subplot(grid[row,column]);a=np.load(DATA/f'patient_surface_ep_w{width:g}_{method}_dt{dt:g}.npz')['activation']
            finite=np.all(np.isfinite(a[view.triangles]),axis=1)
            vals=np.zeros(len(view.triangles));vals[finite]=a[view.triangles[finite]].mean(axis=1)
            colours=cmap(norm(vals));colours[~finite]=mpl.colors.to_rgba('#cfcfcf')
            coll=PolyCollection(view.coordinates[view.triangles[view.order],:2],facecolors=colours[view.order],edgecolors='none',linewidths=0,antialiaseds=False,rasterized=True)
            ax.add_collection(coll)
            for mask,colour,lw in ((z['evaluation_mask'],'#202020',.75),(z['stimulus_mask'],'#00ffff',1.1)):
                seg=_contour_segments(view,mask)
                ax.add_collection(LineCollection(seg,colors='white',linewidths=lw+1.1))
                ax.add_collection(LineCollection(seg,colors=colour,linewidths=lw))
            ax.set_xlim(view.limits[:2]);ax.set_ylim(view.coordinates[:,1].min()-6,view.coordinates[:,1].max()+4)
            ax.set_aspect('equal');ax.set_axis_off()
            metric=frame[(frame.width_mm==width)&(frame.method==method)&(frame.dt_ms==dt)].iloc[0]
            ax.set_title(f'{chr(97+row*2+column)}  {method.capitalize()}',loc='left',pad=2)
            ax.text(.01,.03,f'Distal activation: {100*metric.target_activation_area_fraction:.1f}%',transform=ax.transAxes,fontsize=6.4)
            if column==0:
                ax.text(.0,1.12,f'{width:g}-mm hidden collar',transform=ax.transAxes,fontsize=8,fontweight='bold')
                x0=view.limits[0]+3;y0=view.coordinates[:,1].min()+2
                ax.plot([x0,x0+10],[y0,y0],color='#333',lw=1.2)
                ax.text(x0+5,y0+1,'10 mm',ha='center',fontsize=5.8)
    cax=figure.add_subplot(grid[2,:]);cb=figure.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,50,100,150,210])
    cb.set_label('First activation time (ms)',labelpad=2)
    figure.text(.5,-.014,'Grey: no activation by 210 ms; black: hidden collar; cyan: pacing',ha='center',fontsize=6.2)
    figure.savefig(ROOT/'figures/patient_surface_ep.pdf',bbox_inches='tight')
    figure.savefig(ROOT/'figures/patient_surface_ep.png',dpi=400,bbox_inches='tight')
    plt.close(figure)

if __name__=='__main__': main()
