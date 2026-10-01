from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from model import EPParameters
from run_patient_surface_ep import geometry
from surface_ep import solve_surface_ep


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    start = time.perf_counter()
    input_paths = [DATA / 'patient_extension' / f'P3_left_pair_w{width}_h0_dt0p01.npz'
                   for width in (3, 10)]
    records = []
    for path in input_paths:
        with np.load(path, allow_pickle=False) as source:
            for method in ('screened', 'passive', 'graph'):
                state = source[method + '_state']
                records.append({'input': path.relative_to(ROOT).as_posix(), 'method': method,
                                'minimum': float(state.min()), 'maximum': float(state.max()),
                                'vertices_outside_minus1_plus1': int(np.count_nonzero((state < -1) | (state > 1)))})
    with np.load(input_paths[0], allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    mesh, distance, stimulus, target = geometry(data)
    signed = data['graph_state']
    roundtrip = 2 * (.5 * (1 + np.clip(signed, -1, 1))) - 1
    dt = .025
    baseline_path = DATA / f'patient_surface_ep_w3_graph_dt{dt:g}.npz'
    with np.load(DATA / 'patient_surface_ep_geometry_w3.npz', allow_pickle=False) as recorded:
        if not (np.array_equal(recorded['graph_state'], signed)
                and np.array_equal(recorded['stimulus_mask'], stimulus)
                and np.array_equal(recorded['target_mask'], target)):
            raise RuntimeError('Fresh direct-state output geometry differs from the roundtrip control input.')
    with np.load(baseline_path, allow_pickle=False) as baseline:
        direct = {key: baseline[key].copy() for key in ('v', 'h', 'activation')}
    solution = solve_surface_ep(mesh, roundtrip, dt=dt, t_end=210., stimulus_mask=stimulus,
                                snapshot_times=())
    finite = np.isfinite(direct['activation']) & np.isfinite(solution.activation)
    control = {
        'input': input_paths[0].relative_to(ROOT).as_posix(), 'method': 'graph',
        'dt_ms': dt, 't_end_ms': 210.,
        'phase_min': float(signed.min()), 'phase_max': float(signed.max()),
        'clipped_vertices': sum(item['vertices_outside_minus1_plus1'] for item in records),
        'roundtrip_formula': '2*(0.5*(1+clip(u,-1,1)))-1',
        'maximum_input_roundtrip_difference': float(np.max(np.abs(roundtrip-signed))),
        'maximum_voltage_difference': float(np.max(np.abs(solution.v-direct['v']))),
        'maximum_gate_difference': float(np.max(np.abs(solution.h-direct['h']))),
        'maximum_activation_difference_ms': float(np.max(np.abs(solution.activation[finite]-direct['activation'][finite]))) if finite.any() else 0.,
        'activation_status_disagreements': int(np.count_nonzero(np.isfinite(solution.activation) != np.isfinite(direct['activation']))),
        'common_activated_vertices': int(finite.sum()),
        'baseline_path': baseline_path.relative_to(ROOT).as_posix(),
        'baseline_sha256': digest(baseline_path),
        'input_sha256': digest(input_paths[0]),
        'solver_diagnostics': solution.diagnostics,
    }
    output = DATA / 'patient_surface_ep_transfer_roundtrip.npz'
    np.savez_compressed(output, signed_roundtrip=roundtrip, v=solution.v, h=solution.h,
                        activation=solution.activation)
    control['roundtrip_output'] = output.relative_to(ROOT).as_posix()
    control['roundtrip_output_sha256'] = digest(output)
    control['elapsed_seconds'] = time.perf_counter()-start
    (DATA / 'patient_surface_ep_transfer_control.json').write_text(json.dumps(control, indent=2)+'\n')
    parameter = EPParameters(dt=dt)
    refined_metadata = json.loads((DATA / 'patient_surface_ep_w3_graph_h1_dt0.025.json').read_text())
    protocol = {
        'purpose': 'exploratory reconstruction-to-EP linkage on the pre-existing representative P3 left atrium',
        'patient_selection': 'P3 is the previously selected Experiment 6 representative',
        'input_paths_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in input_paths},
        'code_paths_sha256': {f'code/{name}': digest(ROOT/'code'/name) for name in
                              ('model.py', 'surface_ep.py', 'run_patient_surface_ep.py', 'run_patient_surface_ep_provenance.py')},
        'geometry': {'original_vertices': mesh.n_vertices, 'original_triangles': mesh.n_triangles,
                     'refined_vertices': int(refined_metadata['n_vertices']),
                     'refined_triangles': int(refined_metadata['n_triangles']),
                     'same_piecewise_planar_surface': True},
        'pacing': {'source': 'officially labelled LSPV cut boundary',
                   'distance': 'shortest path on original mesh edge graph; Euclidean edge lengths in mm',
                   'stimulus_collar_mm': 2., 'stimulus_amplitude': 1.2, 'duration_ms': 2.,
                   'distal_distance_mm': 20., 'horizon_ms': 210.,
                   'initial_voltage': 0., 'initial_gate': 1.,
                   'activation_threshold': parameter.activation_threshold,
                   'descriptive_target_capture_fraction': .9},
        'coefficient': {'isotropic_d0_mm2_per_ms': .128,
                        'd0_choice': 'geometric mean sqrt(.32*.0512), without fitting or inferred fibres',
                        'transfer': 'uncensored u enters .001+.999/(1+exp(8*u))',
                        'saturation': False, 'reference_state': '2*reference_score-1'},
        'kinetics': 'unchanged EPParameters Mitchell-Schaeffer kinetics; implicit P1 lumped-mass diffusion, explicit ionic voltage, exact frozen-new-voltage gate',
        'dt_ms_final': [.025, .0125],
        'spatial_control': 'Midpoint subdivision of the same polyhedral surface; prolong original P1 nodal diffusivity and stimulus; evaluate original vertices with original area weights.',
        'activation_support': 'All-four-method intersection for native comparisons; pairwise supports and coverage; common native/refined reference/graph intersection for spatial-control errors.',
        'roundtrip_removal_control': control,
        'phase_extrema': records,
        'limitations': ['Voltage-derived references are not independent electrical-block measurements.',
                        'Absolute patient EP results remain spatially sensitive.',
                        'Unmodified P1 stiffness need not be an M-matrix; no discrete maximum principle is asserted.'],
    }
    (DATA / 'patient_surface_ep_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({name: control[name] for name in (
        'clipped_vertices', 'maximum_input_roundtrip_difference', 'maximum_voltage_difference',
        'maximum_gate_difference', 'maximum_activation_difference_ms',
        'activation_status_disagreements', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
