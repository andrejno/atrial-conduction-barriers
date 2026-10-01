from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from dataclasses import asdict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, Wedge
import numpy as np
import pandas as pd

from model import solve_monodomain
from run_applied import (
    CENTER, LENGTH, RADIUS, TARGET_SECTOR_HALF_ANGLE,
    STIMULUS_RADIUS, STIMULUS_AMPLITUDE, STIMULUS_DURATION,
    _disk_stimulus, analytic_subcell_diffusivity, ep_parameters,
    reconstruction_grid, score_to_fv_diffusivity,
)
from run_reconstruction_ep import target_mask

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MEDIA = ROOT / "media"
CACHE = ROOT / "media_data"
VERSION = "rings-2026-09-30-v1"
BG, PANEL = "#08111f", "#0d1b2b"
WHITE, MUTED = "#edf4fc", "#91a5bd"
COLORS = ("#57d9d1", "#ffbc67")
TIMES = np.arange(211, dtype=float)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases() -> list[dict]:
    table_path = DATA / "reconstruction_ep_topology_capacity.csv"
    table = pd.read_csv(table_path)
    selected = table.loc[
        (table.study == "boundary_production")
        & (table.geometry == "complete_ring")
        & (table.blackout_arc_width_mm == 2)
        & (table.direction == "exit")
    ]
    cases = []
    for method, title in (("reference", "Reference barrier"), ("graph", "Graph reconstruction")):
        row = selected.loc[selected.method == method].iloc[0]
        source = ROOT / row.source_file
        with np.load(source, allow_pickle=False) as saved:
            score = saved["truth" if method == "reference" else method].copy()
        eta = score_to_fv_diffusivity(reconstruction_grid(score.shape[0]), score, 121, subcells=3)
        cases.append(dict(
            name=f"rings_closed_{method}", title=title, group="closed",
            source=str(source.relative_to(ROOT)), source_sha256=_hash(source),
            table=str(table_path.relative_to(ROOT)), checkpoint=row.checkpoint,
            angle_deg=0.0, score=score, diffusivity=eta,
            normalized_capacity=float(row.normalized_capacity),
            reference_arrival=float(row.crossing_time_ms),
            reference_fraction=float(row.target_activated_fraction),
            thresholded_gaps=int(row.gap_count),
            field_key="truth" if method == "reference" else method,
        ))
    table_path = DATA / "pvi_bidirectional_ep.csv"
    table = pd.read_csv(table_path)
    selected = table.loc[(table.experiment == "matched_capacity") & (table.pacing_direction == "exit")]
    for label in ("A", "B"):
        row = selected.loc[selected.matched_case == label].iloc[0]
        eta = analytic_subcell_diffusivity(121, row.gap_width_mm, row.gap_diffusivity_fraction, np.deg2rad(row.gap_angle_deg), subcells=3)
        cases.append(dict(
            name=f"rings_matched_{label}", title=f"Case {label}", group="matched",
            source=str(table_path.relative_to(ROOT)), source_sha256=_hash(table_path),
            table=str(table_path.relative_to(ROOT)), checkpoint=None,
            angle_deg=float(row.gap_angle_deg), score=np.empty((0, 0)), diffusivity=eta,
            width_mm=float(row.gap_width_mm), nominal_gap_diffusivity=float(row.gap_diffusivity_fraction),
            normalized_capacity=float(row.normalized_capacity),
            reference_arrival=float(row.crossing_time_ms),
            reference_fraction=float(row.target_activated_fraction),
            field_key=f"matched_{label}",
        ))
    return cases


def _simulate(case: dict, force: bool = False) -> Path:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{case['name']}.npz"
    parameters = ep_parameters(121, dt=0.02, face_average="harmonic")
    signature = hashlib.sha256((case["source_sha256"] + _hash(ROOT / "code/model.py") + _hash(ROOT / "code/run_applied.py") + json.dumps(asdict(parameters), sort_keys=True)).encode()).hexdigest()
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["signature"]) == signature:
                return path
    eta = case["diffusivity"]
    angle = np.deg2rad(case["angle_deg"])
    target = target_mask(121, case["angle_deg"], "exit")
    xy = (np.arange(121) + 0.5) * LENGTH / 121
    position = np.asarray(CENTER) + 21.5 * np.array([np.cos(angle), np.sin(angle)])
    probe = (int(np.argmin(abs(xy - position[0]))), int(np.argmin(abs(xy - position[1]))))
    solution = solve_monodomain(
        np.zeros_like(eta), parameters, 210.0,
        _disk_stimulus(eta.shape, CENTER), trace_points={"target": probe},
        snapshot_times=TIMES[1:], boundary="noflux", diffusivity_scale=eta,
    )
    activation = solution.activation[target]
    fraction = float(np.isfinite(activation).mean())
    arrival = float(np.nanmin(activation)) if fraction >= 0.8 else None
    assert fraction == case["reference_fraction"]
    if np.isfinite(case["reference_arrival"]):
        assert abs(arrival - case["reference_arrival"]) < 1e-8
    else:
        assert arrival is None
    activation_error = None
    if case["checkpoint"]:
        with np.load(ROOT / case["checkpoint"], allow_pickle=False) as original:
            np.testing.assert_allclose(solution.activation, original["activation"], rtol=0, atol=1e-10, equal_nan=True)
            np.testing.assert_allclose(eta, original["diffusivity"], rtol=0, atol=1e-14)
            activation_error = float(np.nanmax(np.abs(solution.activation - original["activation"])))
    voltage = np.stack([np.zeros_like(eta)] + [solution.snapshots[float(t)] for t in TIMES[1:]]).astype(np.float32)
    target_fraction = np.array([np.mean(np.isfinite(activation) & (activation <= t)) for t in TIMES])
    outcome = dict(
        first_arrival_ms=arrival, target_activated_fraction=fraction,
        captured_by_210_ms=bool(fraction >= 0.8),
        maximum_activation_difference_from_checkpoint_ms=activation_error,
        **solution.diagnostics,
    )
    metadata = {key: value for key, value in case.items() if key not in {"score", "diffusivity", "reference_arrival"}}
    metadata.update(
        parameters=asdict(parameters), pacing_direction="exit", boundary="noflux",
        stimulus=dict(radius_mm=STIMULUS_RADIUS, amplitude=STIMULUS_AMPLITUDE, duration_ms=STIMULUS_DURATION),
        grid_n=121, length_mm=LENGTH, t_end_ms=210.0, frame_interval_ms=1.0,
        rerun=True, temporal_interpolation=False, voltage_storage="float32", activation_storage="float64",
        model_sha256=_hash(ROOT / "code/model.py"), driver_sha256=_hash(ROOT / "code/run_applied.py"),
        outcome=outcome,
    )
    np.savez_compressed(
        path, signature=signature, times=TIMES, voltage=voltage,
        activation=solution.activation, peak=solution.peak, diffusivity=eta,
        source_score=case["score"], target_mask=target,
        target_fraction=target_fraction, trace_time=solution.traces["time"],
        trace_target=solution.traces["target"], metadata=json.dumps(metadata, allow_nan=False),
    )
    return path


def _read(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key].copy() for key in saved.files if key not in {"metadata", "signature"}}
        result["meta"] = json.loads(str(saved["metadata"]))
    return result


def _figure(records: list[dict], group: str):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": WHITE, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#34445a", "savefig.facecolor": BG,
        "mathtext.fontset": "dejavusans", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    title = "A closed contour can still conduct" if group == "closed" else "Nearly equal conductance, different propagation"
    fig.text(0.052, 0.946, title, fontsize=25, weight="semibold")
    detail = "Complete ring · 2 mm boundary-covered blackout" if group == "closed" else "Passive conductance differs by 0.098%"
    fig.text(0.053, 0.907, detail, fontsize=13, color=MUTED)
    clock = fig.text(0.946, 0.921, "0 ms", ha="right", fontsize=24, color=WHITE)
    artists, texts, lines, points = [], [], [], []
    for i, record in enumerate(records):
        left = 0.066 + 0.47 * i
        meta = record["meta"]
        ax = fig.add_axes([left, 0.271, 0.373, 0.60], facecolor=PANEL)
        ax.set_aspect("equal")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        image = ax.imshow(record["voltage"][0].T, origin="lower", extent=(0, 60, 0, 60), cmap="magma", vmin=0, vmax=1, interpolation="bilinear")
        artists.append(image)
        if group == "closed":
            score = record["source_score"]
            coordinate = np.arange(score.shape[0]) * LENGTH / score.shape[0]
            ax.contour(coordinate, coordinate, score.T, levels=[0], colors=[WHITE], linewidths=1.0, alpha=0.65)
            subtitle = "Zero thresholded gaps" if i == 0 else "Zero thresholded gaps · 2 mm blackout"
        else:
            xy = (np.arange(121) + 0.5) * LENGTH / 121
            ax.contour(xy, xy, record["diffusivity"].T, levels=[0.25], colors=[WHITE], linewidths=1.0, alpha=0.65)
            subtitle = f"{meta['width_mm']:g} mm gap · η = {meta['nominal_gap_diffusivity']:g} · {meta['angle_deg']:g}°"
        theta = meta["angle_deg"]
        ax.add_patch(Wedge(CENTER, 23, theta - 15, theta + 15, width=3, fill=False, edgecolor=COLORS[i], linewidth=1.5))
        ax.add_patch(Circle(CENTER, STIMULUS_RADIUS, fill=False, edgecolor=MUTED, linestyle=(0, (2, 3)), linewidth=0.8))
        ax.annotate("", (16, 50), (5, 50), arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
        ax.text(5, 47.7, "fibres", fontsize=9, color=MUTED)
        ax.plot([5, 15], [5, 5], color=WHITE, lw=2)
        ax.text(10, 7, "10 mm", ha="center", fontsize=10, color=WHITE)
        fig.text(left, 0.861, meta["title"], fontsize=19, color=COLORS[i], weight="semibold")
        fig.text(left, 0.829, subtitle, fontsize=11, color=MUTED)
        fraction_label = fig.text(left + 0.373, 0.286, "Target activated  0%", ha="right", fontsize=14, color=COLORS[i], weight="medium")
        texts.append(fraction_label)
        if group == "matched":
            fig.text(left, 0.286, f"C* = {meta['normalized_capacity']:.7f}", fontsize=12, color=MUTED)
        else:
            fig.text(left, 0.286, "White: zero-score contour", fontsize=11, color=MUTED)
    cax = fig.add_axes([0.474, 0.383, 0.012, 0.34])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="magma"), cax=cax)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=10, length=0, pad=5)
    cb.outline.set_visible(False)
    cax.set_title("V", fontsize=13, pad=9)
    trace = fig.add_axes([0.089, 0.098, 0.81, 0.137], facecolor=BG)
    trace.set_xlim(0, 210)
    trace.set_ylim(-3, 105)
    trace.set_xticks([0, 50, 100, 150, 210])
    trace.set_yticks([0, 80, 100], ["0", "80", "100"])
    trace.tick_params(labelsize=10, length=3)
    trace.set_ylabel("Target activated (%)", fontsize=10)
    trace.set_xlabel("Physical time (ms)", fontsize=10, labelpad=7)
    trace.axhline(80, color=MUTED, lw=0.8, linestyle=(0, (3, 4)), alpha=0.7)
    trace.text(212, 80, "capture", fontsize=10, va="center", color=MUTED)
    for i, record in enumerate(records):
        trace.plot(record["times"], 100 * record["target_fraction"], color=COLORS[i], lw=1.0, alpha=0.14)
        line, = trace.plot([], [], color=COLORS[i], lw=2.0, linestyle="-" if i else (0, (4, 2)))
        point, = trace.plot([], [], "o", color=COLORS[i], ms=5)
        lines.append(line)
        points.append(point)
    cursor = trace.axvline(0, color=WHITE, lw=0.8, alpha=0.5)
    fig.text(0.052, 0.018, "Exit pacing · 121 × 121 cells · Δt = 0.02 ms · 210 ms horizon", color=MUTED, fontsize=10)
    legend = "Normalized voltage · marked distal sectors" if group == "closed" else "V normalized · white: η = 0.25 · coloured: distal sector"
    fig.text(0.947, 0.018, legend, color=MUTED, fontsize=10, ha="right")

    def update(index: int):
        index = min(index, len(TIMES) - 1)
        clock.set_text(f"{TIMES[index]:.0f} ms")
        for i, record in enumerate(records):
            artists[i].set_data(record["voltage"][index].T)
            fraction = record["target_fraction"][index]
            texts[i].set_text(f"Target activated  {fraction:.0%}")
            lines[i].set_data(TIMES[:index + 1], 100 * record["target_fraction"][:index + 1])
            points[i].set_data([TIMES[index]], [100 * fraction])
        cursor.set_xdata([TIMES[index], TIMES[index]])
        return artists + texts + lines + points + [cursor, clock]
    return fig, update


def _render(paths: list[Path], group: str, force: bool = False) -> dict:
    MEDIA.mkdir(exist_ok=True)
    stem = MEDIA / ("rings_closed_contour" if group == "closed" else "rings_matched_conductance")
    products = {ext: stem.with_suffix(f".{ext}") for ext in ("mp4", "gif", "png", "json")}
    render_signature = hashlib.sha256((VERSION + _hash(Path(__file__)) + "".join(_hash(path) for path in paths)).encode()).hexdigest()
    if not force and all(path.exists() for path in products.values()):
        saved = json.loads(products["json"].read_text())
        output_hashes = {ext: _hash(path) for ext, path in products.items() if ext != "json"}
        if saved.get("render_signature") == render_signature and saved.get("output_sha256") == output_hashes:
            return {key: str(path.relative_to(ROOT)) for key, path in products.items()}
    records = [_read(path) for path in paths]
    with plt.rc_context():
        fig, update = _figure(records, group)
        writer = FFMpegWriter(fps=15, codec="libx264", bitrate=-1, extra_args=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "1"])
        with writer.saving(fig, str(products["mp4"]), dpi=100):
            for index in list(range(211)) + [210] * 24:
                update(index)
                writer.grab_frame(facecolor=BG)
        poster_index = 110 if group == "closed" else 180
        update(poster_index)
        fig.savefig(products["png"], dpi=100, facecolor=BG)
        plt.close(fig)
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(products["mp4"]),
        "-filter_complex", "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0", "-threads", "1", str(products["gif"]),
    ], check=True)
    provenance = dict(
        render_signature=render_signature, title="A closed contour can still conduct" if group == "closed" else "Nearly equal conductance, different propagation",
        source_cases=[record["meta"] for record in records],
        numerical_arrays=[str(path.relative_to(ROOT)) for path in paths],
        dimensions_px=[1600, 900], fps=15, frame_samples_ms=1, duration_seconds=235 / 15,
        temporal_interpolation=False, final_frame_hold_seconds=24 / 15,
        spatial_rendering="bilinear image display of saved finite-volume states",
        units=dict(space="mm", physical_time="ms", voltage="normalized", target_activation="percent"),
        voltage_color_range=[0, 1], voltage_values_above_1="display saturated; numerical arrays unchanged",
        capture_criterion="at least 80 percent of the distal sector activated by 210 ms",
        output_sha256={ext: _hash(path) for ext, path in products.items() if ext != "json"},
    )
    products["json"].write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return {key: str(path.relative_to(ROOT)) for key, path in products.items()}


def generate_all(force: bool = False, rerun: bool = False) -> list[dict]:
    cases = _cases()
    paths = {case["name"]: _simulate(case, force=rerun) for case in cases}
    return [_render([paths[case["name"]] for case in cases if case["group"] == group], group, force=force) for group in ("closed", "matched")]


if __name__ == "__main__":
    print(json.dumps(generate_all(), indent=2))
