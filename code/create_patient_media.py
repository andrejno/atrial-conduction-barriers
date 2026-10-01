from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from patient_surface_plot import anatomical_view
from surface_mesh import TriangleMesh
from surface_ep import solve_surface_ep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
MEDIA = ROOT / 'media'
CACHE = ROOT / 'media_data'
BG = '#08111f'
PANEL = '#0d1b2b'
WHITE = '#edf4fc'
MUTED = '#91a5bd'
CYAN = '#57d9d1'
AMBER = '#ffbc67'
CREDIT = 'P3 anatomy: Martínez Díaz et al., 2024 · CC BY 4.0'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 14,
         'text.color': WHITE, 'axes.labelcolor': WHITE,
         'xtick.color': MUTED, 'ytick.color': MUTED,
         'axes.edgecolor': MUTED, 'savefig.facecolor': BG,
         'figure.facecolor': BG, 'axes.facecolor': BG}


def _read(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _camera(data):
    boundaries = {s: data['boundary_' + s] for s in ('LSPV', 'LIPV', 'RSPV', 'RIPV')}
    view = anatomical_view(data['points'], data['triangles'], boundaries, resolution=400)
    xyz = view.coordinates.copy()
    xyz -= .5 * (xyz.min(axis=0) + xyz.max(axis=0))
    return xyz


def _projection(xyz, triangles, yaw=0., pitch=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    q = xyz @ ry.T @ rx.T
    faces = q[triangles]
    order = np.argsort(faces[:, :, 2].mean(axis=1), kind='stable')
    normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1.e-15)
    lighting = .72 + .28 * np.abs(normal @ np.array([-.25, .35, .9027735]))
    return faces[order, :, :2], order, lighting


def _surface(ax, xyz, triangles, values, *, cmap='viridis', vmax=1., support=None, yaw=0., pitch=0.):
    vertices, order, light = _projection(xyz, triangles, yaw, pitch)
    colors = plt.get_cmap(cmap)(Normalize(0, vmax, clip=True)(np.mean(values[triangles], axis=1)))
    if support is not None:
        colors[~np.all(support[triangles], axis=1)] = matplotlib.colors.to_rgba('#647185')
    colors[:, :3] *= light[:, None]
    collection = PolyCollection(vertices, facecolors=colors[order], edgecolors='none',
                                linewidths=0, antialiaseds=False)
    ax.add_collection(collection)
    span = .57 * max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2]))
    ax.set(xlim=(-span, span), ylim=(-span, span), aspect='equal')
    ax.axis('off')
    return collection


def _colorbar(fig, rect, cmap, label, *, vmax=1., last_tick=None):
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=cmap),
                     cax=fig.add_axes(rect), orientation='horizontal', ticks=[0, .5 * vmax, vmax])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=12, length=3, color=MUTED)
    if last_tick is not None:
        cb.ax.set_xticklabels(['0', f'{.5 * vmax:g}', last_tick])
    cb.set_label(label, fontsize=13, labelpad=7)


def _frame_layout(title, subtitle):
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.text(.035, .948, title, fontsize=29, weight='bold', va='top')
    fig.text(.035, .894, subtitle, fontsize=15, color=MUTED, va='top')
    fig.text(.035, .025, CREDIT, fontsize=10.5, color=MUTED)
    return fig


def _preview(movie, *, width=800, fps=10):
    gif = movie.with_suffix('.gif')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(movie),
                    '-vf', f'fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];'
                           '[s0]palettegen=max_colors=128:stats_mode=diff[p];'
                           '[s1][p]paletteuse=dither=sierra2_4a', '-loop', '0', str(gif)], check=True)
    return gif


def _rotation():
    data = _read(DATA / 'zenodo_pvi_representative.npz')
    xyz = _camera(data)
    triangles = data['triangles']
    support = data['confidence'] > 0
    observed = np.zeros_like(data['confidence'])
    observed[support] = .5 * (1 + data['forcing'][support] / data['confidence'][support])
    graph = np.clip(.5 * (1 + data['graph_state']), 0, 1)
    fields = [data['bi_mv'], observed, graph]
    cmaps = ['magma', 'viridis', 'viridis']
    fig = _frame_layout('Clinical atrial substrate', 'P3 · Clinical voltage, incomplete observations and reconstruction')
    fig.text(.965, .894, 'Rotating view', fontsize=13, color=CYAN, ha='right', va='top')
    axes, collections = [], []
    for k, title in enumerate(('Clinical voltage', 'Sparse observations', 'Graph reconstruction')):
        x = .015 + k * .325
        ax = fig.add_axes([x, .235, .32, .6])
        fig.text(x + .16, .836, title, ha='center', fontsize=19, weight='medium')
        collections.append(_surface(ax, xyz, triangles, fields[k], cmap=cmaps[k],
                                    support=support if k == 1 else None))
        axes.append(ax)
    _colorbar(fig, [.07, .185, .22, .015], 'magma', 'Bipolar voltage (mV)', last_tick='≥1')
    _colorbar(fig, [.435, .185, .435, .015], 'viridis', 'Voltage-derived barrier score')
    fig.text(.505, .075, 'Grey: unobserved surface', ha='center', fontsize=13, color=MUTED)
    path = MEDIA / 'patient_reconstruction_rotation.mp4'
    writer = FFMpegWriter(fps=24, codec='libx264', bitrate=6000,
                         extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '2'])
    frame_colors = []
    for k in range(3):
        colors = plt.get_cmap(cmaps[k])(Normalize(0, 1, clip=True)(fields[k][triangles].mean(axis=1)))
        if k == 1:
            colors[~np.all(support[triangles], axis=1)] = matplotlib.colors.to_rgba('#647185')
        frame_colors.append(colors)
    with writer.saving(fig, str(path), 100):
        for i in range(288):
            theta = 2 * np.pi * i / 288
            vertices, order, light = _projection(xyz, triangles, np.deg2rad(48) * np.sin(theta),
                                                np.deg2rad(7) * np.sin(2 * theta))
            for collection, base in zip(collections, frame_colors):
                colors = base.copy()
                colors[:, :3] *= light[:, None]
                collection.set_verts(vertices)
                collection.set_facecolors(colors[order])
            writer.grab_frame()
            if i == 0:
                fig.savefig(path.with_suffix('.png'), dpi=100)
    plt.close(fig)
    _preview(path)
    return {'file': str(path.relative_to(ROOT)), 'source': 'data/zenodo_pvi_representative.npz',
            'frames': 288, 'fps': 24, 'duration_seconds': 12,
            'meaning': 'Camera rotation of static stored fields; not evolution in physical or reconstruction time.',
            'voltage_display_range_mV': [0, 1], 'voltage_display_saturation_above_mV': 1,
            'barrier_score_display_range': [0, 1],
            'score_transform': '0.5*(1+u), saturated to [0,1] only for display, as in manuscript',
            'missing_data': 'Any face containing an unsupported vertex is grey.',
            'surface_rendering': 'Original triangles, per-face mean colors and mild orientation lighting; depth-sorted orthographic projection.'}


def _states(force=False):
    rows, cases, validations = [], [], []
    for width, method, name in [(3, 'reference', 'reference'), (3, 'graph', 'graph_w3'), (10, 'graph', 'graph_w10')]:
        source = DATA / f'patient_surface_ep_geometry_w{width}.npz'
        data = _read(source)
        path = CACHE / f'patient_voltage_{name}.npz'
        meta = path.with_suffix('.json')
        if force or not path.exists():
            mesh = TriangleMesh(data['points'], data['triangles'])
            signed = 2 * data['reference_score'] - 1 if method == 'reference' else data['graph_state']
            sol = solve_surface_ep(mesh, signed, dt=.0125, t_end=210.,
                                   stimulus_mask=data['stimulus_mask'],
                                   snapshot_times=tuple(float(i) for i in range(1, 211)))
            times = np.arange(211, dtype=float)
            states = np.vstack([np.zeros(mesh.n_vertices)] + [sol.snapshots[float(i)] for i in range(1, 211)])
            np.savez_compressed(path, time_ms=times, voltage=states, activation=sol.activation,
                                final_v=sol.v, final_h=sol.h)
            meta.write_text(json.dumps(sol.diagnostics, indent=2))
        cached = _read(path)
        original = _read(DATA / f'patient_surface_ep_w{width}_{method}_dt0.0125.npz')
        same = np.array_equal(np.isfinite(cached['activation']), np.isfinite(original['activation']))
        finite = np.isfinite(original['activation'])
        error = float(np.max(np.abs(cached['activation'][finite] - original['activation'][finite])))
        v_error = float(np.max(np.abs(cached['final_v'] - original['v'])))
        if not same or error > 1.e-7 or v_error > 1.e-7:
            raise RuntimeError(f'{name}: dense-snapshot replay differs from original output')
        target = data['target_mask'].astype(bool)
        area = data['vertex_area']
        fractions = np.array([area[target & (cached['activation'] <= t)].sum() / area[target].sum()
                              for t in cached['time_ms']])
        for t, f in zip(cached['time_ms'], fractions):
            rows.append({'case': name, 'time_ms': t, 'target_activated_area_fraction': f})
        validations.append({'case': name, 'source_geometry': str(source.relative_to(ROOT)),
                            'reference_output': f'data/patient_surface_ep_w{width}_{method}_dt0.0125.npz',
                            'same_activated_vertices': bool(same),
                            'max_activation_difference_ms': error, 'max_final_voltage_difference': v_error})
        cases.append((name, data, cached, fractions))
    pd.DataFrame(rows).to_csv(CACHE / 'patient_target_activation_series.csv', index=False)
    return cases, validations


def _propagation(cases):
    xyz = _camera(cases[0][1])
    triangles = cases[0][1]['triangles']
    fig = _frame_layout('Simulated voltage on clinical anatomy', 'P3 · Reference substrate and reconstruction from incomplete voltage maps')
    clock = fig.text(.965, .894, 't = 0 ms', ha='right', va='top', fontsize=18, color=CYAN)
    labels = ['Reference', '3 mm missing band', '10 mm missing band']
    colors = [WHITE, CYAN, AMBER]
    collections = []
    _, order, light = _projection(xyz, triangles)
    for k, label in enumerate(labels):
        x = .015 + k * .325
        ax = fig.add_axes([x, .355, .32, .485])
        fig.text(x + .16, .836, label, ha='center', fontsize=19, color=colors[k])
        collections.append(_surface(ax, xyz, triangles, cases[k][2]['voltage'][0], cmap='magma'))
    _colorbar(fig, [.775, .312, .18, .012], 'magma', 'Simulated voltage (normalised)', last_tick='≥1')
    ax = fig.add_axes([.245, .13, .465, .19])
    ax.set(xlim=(0, 210), ylim=(0, 100), xlabel='Time (ms)', ylabel='Activated target area (%)')
    ax.tick_params(labelsize=11)
    ax.set_xticks([0, 50, 100, 150, 210]); ax.set_yticks([0, 50, 100])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15, color=MUTED)
    traces = [ax.plot([], [], color=c, lw=2.5)[0] for c in colors]
    cursor = ax.axvline(0, color=MUTED, lw=1, alpha=.7)
    fig.text(.035, .255, 'LSPV pacing', fontsize=17, color=WHITE, va='top')
    fig.text(.035, .21, 'Distal target\n≥20 mm from LSPV', fontsize=13, color=MUTED, linespacing=1.5, va='top')
    fig.text(.765, .185, 'Exploratory forward simulation', fontsize=13, color=MUTED)
    value_texts = [fig.text(.765, .15 - .029 * k, '', color=c, fontsize=12.5) for k, c in enumerate(colors)]
    path = MEDIA / 'patient_voltage_propagation.mp4'
    writer = FFMpegWriter(fps=15, codec='libx264', bitrate=6000,
                         extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '2'])
    frame_ids = [0] * 8 + list(range(211)) + [210] * 15
    with writer.saving(fig, str(path), 100):
        for fi, idx in enumerate(frame_ids):
            for k, (collection, case) in enumerate(zip(collections, cases)):
                values = case[2]['voltage'][idx][triangles].mean(axis=1)
                rgba = plt.get_cmap('magma')(Normalize(0, 1, clip=True)(values))
                rgba[:, :3] *= light[:, None]
                collection.set_facecolors(rgba[order])
                traces[k].set_data(case[2]['time_ms'][:idx+1], 100 * case[3][:idx+1])
                value_texts[k].set_text(f'{labels[k]}   {100 * case[3][idx]:.1f}%')
            clock.set_text(f't = {idx:d} ms')
            cursor.set_xdata([idx, idx])
            writer.grab_frame()
            if idx == 155 and fi == 163:
                fig.savefig(path.with_suffix('.png'), dpi=100)
    plt.close(fig)
    _preview(path)
    return {'file': str(path.relative_to(ROOT)), 'fps': 15, 'frames': len(frame_ids),
            'duration_seconds': len(frame_ids) / 15, 'physical_time_ms': [0, 210],
            'solver': 'code/surface_ep.py:solve_surface_ep', 'dt_ms': .0125,
            'snapshot_interval_ms': 1., 'displayed_voltage_range': [0, 1], 'voltage_units': 'dimensionless',
            'display_saturation': 'Voltage >1 is assigned the upper color only; solver states are not clipped.',
            'model': 'Original isotropic P1 surface monodomain model, original logistic diffusivity and kinetics.',
            'boundaries': 'Original open P3 mesh; natural zero flux on every opening; no anatomical caps added.',
            'stimulus': 'Original LSPV distance <=2 mm, amplitude 1.2, duration 2 ms.',
            'target': 'Original geodesic distance from LSPV >=20 mm, area-weighted activation fraction.',
            'scope': 'Exploratory forward simulations on clinical anatomy, not measured activation or patient-specific predictions.',
            'case_selection': ['reference', 'graph reconstruction with 3 mm left-PV blackout',
                               'graph reconstruction with 10 mm left-PV blackout']}


def _digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _input_signature():
    paths = [Path(__file__), ROOT / 'DATA_ATTRIBUTION.md']
    paths += [ROOT / 'code' / name for name in
              ('surface_ep.py', 'surface_mesh.py', 'surface_fem.py', 'model.py', 'patient_surface_plot.py')]
    paths += [DATA / name for name in ('zenodo_pvi_representative.npz',
              'patient_surface_ep_geometry_w3.npz', 'patient_surface_ep_geometry_w10.npz',
              'patient_surface_ep_w3_reference_dt0.0125.npz',
              'patient_surface_ep_w3_graph_dt0.0125.npz', 'patient_surface_ep_w10_graph_dt0.0125.npz')]
    return {str(path.relative_to(ROOT)): _digest(path) for path in paths}


def _output_paths():
    paths = [MEDIA / (name + suffix) for name in
             ('patient_reconstruction_rotation', 'patient_voltage_propagation')
             for suffix in ('.mp4', '.gif', '.png')]
    paths += [CACHE / ('patient_voltage_' + name + suffix)
              for name in ('reference', 'graph_w3', 'graph_w10') for suffix in ('.npz', '.json')]
    paths.append(CACHE / 'patient_target_activation_series.csv')
    return paths


def generate_all(force=False):
    MEDIA.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    record = CACHE / 'patient_media_provenance.json'
    signature = _input_signature()
    movies = [MEDIA / 'patient_reconstruction_rotation.mp4', MEDIA / 'patient_voltage_propagation.mp4']
    if not force and record.exists():
        previous = json.loads(record.read_text())
        if previous.get('input_sha256') == signature and all(
            path.exists() and previous.get('artifact_sha256', {}).get(str(path.relative_to(ROOT))) == _digest(path)
            for path in _output_paths()
        ):
            return movies
    cases, validations = _states(force=force)
    with plt.rc_context(STYLE):
        rotation = _rotation()
        propagation = _propagation(cases)
    provenance = {'source_credit': CREDIT, 'source_doi': '10.5281/zenodo.10726677',
                  'attribution_file': 'DATA_ATTRIBUTION.md', 'resolution': [1600, 900],
                  'rotation': rotation, 'propagation': propagation, 'replay_validation': validations,
                  'input_sha256': signature,
                  'artifact_sha256': {str(path.relative_to(ROOT)): _digest(path) for path in _output_paths()}}
    record.write_text(json.dumps(provenance, indent=2))
    return movies


if __name__ == '__main__':
    for path in generate_all():
        print(path)
