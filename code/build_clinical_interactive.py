from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def packed(values, dtype):
    a = np.asarray(values, dtype=dtype)
    return {'dtype': a.dtype.name, 'shape': list(a.shape), 'data': base64.b64encode(a.tobytes()).decode()}


REPOSITORY = Path(__file__).resolve().parents[1]
REQUIRED = (
    'data/patient_surface_ep_geometry_w3.npz',
    'data/patient_surface_ep_geometry_w10.npz',
    'data/patient_extension/patient_extension_patient_summary.csv',
    'data/patient_extension/patient_extension_patient_capacity.csv',
)


def select_source(source):
    root = Path(source).resolve()
    if not root.exists() and root.name == 'source':
        parent = root.parent
        if (parent / 'repository_inputs.json').is_file() or (parent / 'latest_run.json').is_file():
            root = parent
    candidates = []
    pointer = root / 'latest_run.json'
    if pointer.is_file():
        recorded = Path(json.loads(pointer.read_text())['run'])
        if recorded.is_absolute() or '..' in recorded.parts:
            raise ValueError('latest_run.json must identify a relative run inside the source directory.')
        candidate = (root / recorded).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError('Recorded run lies outside the source directory.')
        candidates.append(candidate)
    runs = []
    for status in (root / 'runs').glob('*/execution_status.json'):
        directory = status.parent.resolve()
        if not directory.is_relative_to(root):
            continue
        metadata = json.loads(status.read_text())
        if metadata.get('status') == 'completed':
            runs.append((metadata.get('finished_utc', directory.name), directory))
    candidates += [path for _, path in sorted(runs, reverse=True)]
    candidates.append(root)
    for directory in dict.fromkeys(candidates):
        if not all((directory / name).is_file() for name in REQUIRED):
            continue
        status = directory / 'execution_status.json'
        metadata = json.loads(status.read_text()) if status.is_file() else {}
        if status.is_file() and metadata.get('status') != 'completed':
            continue
        for name in REQUIRED:
            expected = metadata.get('results_sha256', {}).get(name)
            if expected and hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Completed clinical result changed: {name}')
        return directory, {
            'kind': 'completed_run' if status.is_file() else 'stored_source_data',
            'relative_run': directory.relative_to(root).as_posix(),
            'selection': 'Recorded latest completed run, then latest available complete run, then source data.',
        }
    raise FileNotFoundError('Clinical inputs are incomplete. Run python code/reproduce.py first, or pass --source PATH containing the completed research data.')


def build(source, output):
    source, selection = select_source(source)
    paths = [source/'data'/f'patient_surface_ep_geometry_w{w}.npz' for w in (3,10)]
    data = [dict(np.load(p)) for p in paths]
    a, b = data
    assert np.array_equal(a['points'],b['points']) and np.array_equal(a['triangles'],b['triangles'])
    assert np.array_equal(a['reference_score'],b['reference_score'])
    points, tri = a['points'], a['triangles']
    centres={s: points[a['boundary_'+s]].mean(axis=0) for s in ('LSPV','LIPV','RSPV','RIPV')}
    def unit(x): return x/np.linalg.norm(x)
    right=unit(centres['RSPV']+centres['RIPV']-centres['LSPV']-centres['LIPV'])
    up=centres['LSPV']+centres['RSPV']-centres['LIPV']-centres['RIPV']
    up=unit(up-np.dot(up,right)*right); front=unit(np.cross(right,up))
    if np.dot(front,np.mean(list(centres.values()),axis=0)-points.mean(axis=0))<0: front=-front
    xyz=(points-points.mean(axis=0))@np.column_stack([right,up,front])
    xyz-=.5*(xyz.min(axis=0)+xyz.max(axis=0))
    face=np.cross(xyz[tri[:,1]]-xyz[tri[:,0]],xyz[tri[:,2]]-xyz[tri[:,0]])
    normals=np.zeros_like(xyz)
    for k in range(3):np.add.at(normals,tri[:,k],face)
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
    fields={'voltage':a['bi_mv'],'reference':a['reference_score']}
    payload={'positions':packed(xyz,'<f4'),'normals':packed(normals,'<f4'),'triangles':packed(tri,'<u2')}
    for w,z in zip((3,10),data):
        support=z['confidence']>0
        observed=np.zeros_like(z['confidence'])
        observed[support]=.5*(1+z['forcing'][support]/z['confidence'][support])
        fields[f'graph{w}']=np.clip(.5*(1+z['graph_state']),0,1)
        fields[f'observed{w}']=observed
        payload[f'support{w}']=packed(support,'u1')
        payload[f'blackout{w}']=packed(z['blackout_mask'],'u1')
    payload['fields']={name:packed(values,'<f4') for name,values in fields.items()}
    payload['vertexCount']=len(xyz);payload['triangleCount']=len(tri)
    metrics_path=source/'data/patient_extension/patient_extension_patient_summary.csv'
    capacity_path=source/'data/patient_extension/patient_extension_patient_capacity.csv'
    metrics=pd.read_csv(metrics_path); capacity=pd.read_csv(capacity_path)
    metrics=metrics[(metrics.mesh_level==0)&np.isclose(metrics.pseudo_dt,.01)&(metrics.metric_support=='common_3mm')]
    capacity=capacity[(capacity.mesh_level==0)&np.isclose(capacity.pseudo_dt,.01)]
    keys=['patient_id','method','evaluation_width_mm']
    joined=metrics[keys+['rmse_area_weighted','dice_voltage_le_0p1','observation_area_fraction']].merge(
        capacity[keys+['capacity_absolute_error']],on=keys,validate='one_to_one').sort_values(keys)
    assert len(joined)==54
    payload['metrics']=joined.to_dict(orient='records')
    import matplotlib
    payload['palettes']={name:(matplotlib.colormaps[name](np.linspace(0,1,256))[:,:3]*255).round().astype(int).tolist() for name in ('magma','viridis')}
    output.mkdir(parents=True,exist_ok=True)
    (output/'clinical-data.js').write_text('window.CLINICAL_DATA='+json.dumps(payload,separators=(',',':'))+';\n')
    joined.to_csv(output/'clinical-metrics.csv',index=False)
    source_paths=paths+[metrics_path,capacity_path]
    provenance={'source_selection':selection,'source_doi':'10.5281/zenodo.10726677','licence':'CC BY 4.0',
        'citation':'Martínez Díaz et al. (2024), Atrial Models with Personalized Effective Refractory Period, version 1.0.',
        'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'mesh':{'patient':'P3','vertices':len(xyz),'triangles':len(tri),'units':'mm','topology':'Original source-derived open P3 triangular surface; no additional caps or geometry.'},
        'fields':{'voltage':'Interpolated source clinical bipolar voltage, mV; display saturated at1mV only.',
            'reference':'Original threshold-anchored score b(V)=2^(-V/0.1mV).',
            'graph3':'Stored graph reconstruction, 3mm left-PV blackout; score=clip((1+u)/2,0,1).',
            'graph10':'Stored graph reconstruction,10mm left-PV blackout; score=clip((1+u)/2,0,1).',
            'observed3':'Forcing/confidence converted to score at observed nodes; any triangle with an unsupported node is displayed grey.',
            'observed10':'Same definition with10mm blackout.'},
        'rendering':'Offline WebGL with actual indexed mesh, linearly interpolated nodal values, and mild two-sided lighting. Canvas fallback uses triangle means. Pointer values use barycentric interpolation on the foremost triangle.',
        'quantisation':'Positions, normals and scalar fields stored asFloat32 for browser delivery; triangle indices unchanged UInt16.',
        'withheld_outline':'Blackout-mask boundary constructed at midpoints of mesh edges crossing the stored Boolean mask.',
        'metrics':'54 stored patient-summary records:6patients×3methods×3widths; mesh_level=0,pseudo_dt=.01. RMSE and Dice evaluated on fixed common3mm support. Patient summaries average left/right PV withholding experiments. Capacity errors use the fixed analysis domain.',
        'patient_operator':'Isotropic surface reconstruction and isotropic surface EP; no patient fibre field inferred.',
        'scope':'Stored reconstruction fields, not a live parameterized PDE solution. Clinical EP discussed alongside these fields is exploratory forward simulation, not measured activation.',
        'numerical_integrity':{'source_arrays_changed':False,'mesh_decimation':False,'new_fitted_models':False}}
    (output/'clinical-provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print(output/'clinical-data.js', (output/'clinical-data.js').stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=REPOSITORY/'.research',help='Research workspace (default: repository .research); selects the latest completed run.')
    p.add_argument('--output',type=Path,default=REPOSITORY/'docs/interactive',help='Interactive asset directory (default: repository docs/interactive).')
    args=p.parse_args();build(args.source,args.output)
