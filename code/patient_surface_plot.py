"""Anatomical, orthographic plots for the patient-surface experiment.

The camera uses only the four officially labelled PV openings.  Every panel
shares this camera, triangle depth order, limits and score scale.  A small
z-buffer hides contacts and contour segments behind the visible surface;
posterior surface details are therefore not painted through the atrium.
The displayed contacts are deterministic subsamples of the supplied dense
clinical map, not original catheter acquisition points.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection


PATIENT_IDS = ("P1", "P3", "P4", "P5", "P6", "P7")
METHOD_COLOURS = {"screened": "#777777", "passive": "#2878a3", "graph": "#b83852"}
MASK_COLOUR = "#542e7b"


@dataclass
class OrthographicView:
    coordinates: np.ndarray
    triangles: np.ndarray
    order: np.ndarray
    limits: tuple[float, float, float, float]
    depth_buffer: np.ndarray
    pixel_step: float


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("PV landmarks do not define a nondegenerate anatomical camera")
    return vector / norm


def anatomical_view(
    points: np.ndarray,
    triangles: np.ndarray,
    boundaries: Mapping[str, np.ndarray],
    *,
    resolution: int = 1000,
) -> OrthographicView:
    """Return a posterior orthographic camera and a visibility z-buffer.

    The horizontal direction joins the left and right PV pairs.  The vertical
    direction joins inferior and superior pairs after orthogonalization.  The
    viewing side is the side of the PV landmarks relative to the mesh centroid.
    No voltage, reconstruction or error field enters the camera construction.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    centres = {label: points[np.asarray(boundaries[label], dtype=int)].mean(axis=0)
               for label in ("LSPV", "LIPV", "RSPV", "RIPV")}
    lateral = _unit(centres["RSPV"] + centres["RIPV"] - centres["LSPV"] - centres["LIPV"])
    superior = centres["LSPV"] + centres["RSPV"] - centres["LIPV"] - centres["RIPV"]
    superior = _unit(superior - np.dot(superior, lateral) * lateral)
    toward_viewer = _unit(np.cross(lateral, superior))
    centre = points.mean(axis=0)
    pv_offset = np.mean(list(centres.values()), axis=0) - centre
    if np.dot(toward_viewer, pv_offset) < 0.0:
        toward_viewer = -toward_viewer
    xyz = (points - centre) @ np.column_stack((lateral, superior, toward_viewer))
    extent = np.ptp(xyz[:, :2], axis=0)
    side = 1.12 * float(np.max(extent))
    middle = 0.5 * (xyz[:, :2].min(axis=0) + xyz[:, :2].max(axis=0))
    limits = (middle[0] - side / 2, middle[0] + side / 2,
              middle[1] - side / 2, middle[1] + side / 2)
    order = np.argsort(xyz[triangles, 2].mean(axis=1), kind="stable")
    depth = np.full((resolution, resolution), -np.inf)
    pixel_step = side / (resolution - 1)
    # A barycentric z-buffer is used only for visibility of overlays.  The
    # coloured surface itself remains a depth-sorted vector collection.
    for nodes in triangles:
        face = xyz[nodes]
        px = (face[:, 0] - limits[0]) / pixel_step
        py = (face[:, 1] - limits[2]) / pixel_step
        x0, x1 = max(0, int(np.floor(px.min()))), min(resolution - 1, int(np.ceil(px.max())))
        y0, y1 = max(0, int(np.floor(py.min()))), min(resolution - 1, int(np.ceil(py.max())))
        if x1 < x0 or y1 < y0:
            continue
        determinant = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2])
        if abs(determinant) < 1e-10:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        a = ((py[1] - py[2]) * (xx - px[2]) + (px[2] - px[1]) * (yy - py[2])) / determinant
        b = ((py[2] - py[0]) * (xx - px[2]) + (px[0] - px[2]) * (yy - py[2])) / determinant
        c = 1.0 - a - b
        inside = (a >= -1e-10) & (b >= -1e-10) & (c >= -1e-10)
        z = a * face[0, 2] + b * face[1, 2] + c * face[2, 2]
        tile = depth[y0:y1 + 1, x0:x1 + 1]
        np.maximum(tile, np.where(inside, z, -np.inf), out=tile)
    return OrthographicView(xyz, triangles, order, limits, depth, pixel_step)


def _visible(view: OrthographicView, positions: np.ndarray) -> np.ndarray:
    pixels = np.rint((positions[:, :2] - [view.limits[0], view.limits[2]]) / view.pixel_step).astype(int)
    pixels = np.clip(pixels, 0, view.depth_buffer.shape[0] - 1)
    front_depth = view.depth_buffer[pixels[:, 1], pixels[:, 0]]
    # A sub-millimetre tolerance compensates raster sampling on steep faces;
    # it is small compared with the separation of the two atrial walls.
    return positions[:, 2] >= front_depth - 0.35


def _contour_segments(view: OrthographicView, mask: np.ndarray) -> np.ndarray:
    triangles = view.triangles
    mask = np.asarray(mask, dtype=bool)
    mixed = np.any(mask[triangles], axis=1) & ~np.all(mask[triangles], axis=1)
    segments = []
    for nodes in triangles[mixed]:
        intersections = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if mask[nodes[a]] != mask[nodes[b]]:
                intersections.append(0.5 * (view.coordinates[nodes[a]] + view.coordinates[nodes[b]]))
        if len(intersections) == 2:
            segment = np.asarray(intersections)
            if _visible(view, segment.mean(axis=0, keepdims=True))[0]:
                segments.append(segment[:, :2])
    return np.asarray(segments).reshape(-1, 2, 2)


def surface_panel(
    axis: plt.Axes,
    view: OrthographicView,
    values: np.ndarray,
    *,
    title: str,
    support: np.ndarray | None = None,
    evaluation: np.ndarray | None = None,
    contacts: np.ndarray | None = None,
    landmarks: Mapping[str, np.ndarray] | None = None,
) -> mpl.cm.ScalarMappable:
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(0.0, 1.0)
    face_values = np.mean(np.asarray(values)[view.triangles], axis=1)
    colours = cmap(norm(face_values))
    if support is not None:
        # A face crossing unsupported nodes has no fully observed interpolant.
        # It is grey rather than filling absent data with an arbitrary score.
        colours[~np.all(np.asarray(support, dtype=bool)[view.triangles], axis=1)] = mpl.colors.to_rgba("#dadada")
    collection = PolyCollection(
        view.coordinates[view.triangles[view.order], :2],
        facecolors=colours[view.order], edgecolors="none", linewidths=0,
        antialiaseds=False, rasterized=True,
    )
    axis.add_collection(collection)
    if evaluation is not None:
        segments = _contour_segments(view, evaluation)
        axis.add_collection(LineCollection(segments, colors="white", linewidths=1.5))
        axis.add_collection(LineCollection(segments, colors=MASK_COLOUR, linewidths=0.8))
    if contacts is not None:
        positions = view.coordinates[np.asarray(contacts, dtype=int)]
        positions = positions[_visible(view, positions)]
        axis.scatter(positions[:, 0], positions[:, 1], s=2.7, c="#111111", edgecolors="white", linewidths=0.14)
    if landmarks is not None:
        for label in ("LSPV", "LIPV", "RSPV", "RIPV"):
            centre = view.coordinates[np.asarray(landmarks[label], dtype=int)].mean(axis=0)
            axis.text(centre[0], centre[1], label, ha="center", va="center", fontsize=5.8,
                      color="#111111", bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.6))
    axis.set_xlim(view.limits[:2])
    # Crop unused vertical camera space while retaining the common projection;
    # this lets the anatomy use the full width of each publication panel.
    display_ymin = float(view.coordinates[:, 1].min()) - 10.0
    display_ymax = float(view.coordinates[:, 1].max()) + 4.0
    axis.set_ylim(display_ymin, display_ymax)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(title, loc="left", pad=3)
    x0 = view.limits[0]
    sx, sy = x0 + 3.0, display_ymin + 3.0
    axis.plot([sx, sx + 10], [sy, sy], color="#333333", lw=1.5)
    axis.text(sx + 5, sy + 1, "10 mm", ha="center", va="bottom", fontsize=6)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def _style() -> None:
    mpl.rcParams.update({
        "font.size": 8.0, "axes.titlesize": 8.3, "axes.labelsize": 8.0,
        "legend.fontsize": 6.7, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "figure.dpi": 140, "savefig.dpi": 400, "pdf.fonttype": 42,
        "ps.fonttype": 42, "font.family": "serif", "mathtext.fontset": "stix",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def make_patient_surface_figure(
    representative_path: Path,
    summary: pd.DataFrame,
    capacity: pd.DataFrame,
    output_base: Path,
) -> None:
    """Plot recorded fields and patient results; no fitting is performed here."""
    _style()
    with np.load(representative_path) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    boundaries = {label: data[f"boundary_{label}"] for label in ("LSPV", "LIPV", "RSPV", "RIPV")}
    view = anatomical_view(data["points"], data["triangles"], boundaries)
    support = data["confidence"] > 0.0
    observed = np.zeros_like(data["confidence"], dtype=float)
    observed[support] = 0.5 * (1.0 + data["forcing"][support] / data["confidence"][support])
    # Same prespecified readout as the simulation; stored phase states are not
    # altered.  Score saturation is a reporting/transfer map, not a solver step.
    graph_score = np.clip(0.5 * (1.0 + data["graph_state"]), 0.0, 1.0)
    figure = plt.figure(figsize=(7.2, 5.5), layout="constrained")
    grid = figure.add_gridspec(3, 3, height_ratios=(1.05, 0.13, 1.0), hspace=0.10, wspace=0.09)
    top = [figure.add_subplot(grid[0, k]) for k in range(3)]
    bottom = [figure.add_subplot(grid[2, k]) for k in range(3)]
    first = surface_panel(top[0], view, data["reference_score"], title="a  P3 clinical-map score", landmarks=boundaries)
    surface_panel(top[1], view, observed, title="b  Sparse map observations", support=support,
                  evaluation=data["evaluation_mask"], contacts=data["contact_nodes"])
    surface_panel(top[2], view, graph_score, title="c  Graph reconstruction", evaluation=data["evaluation_mask"])
    colour_axis = figure.add_subplot(grid[1, :2])
    colourbar = figure.colorbar(first, cax=colour_axis, orientation="horizontal", ticks=[0, .25, .5, .75, 1])
    colourbar.set_label("Threshold-anchored barrier score", labelpad=2)
    note_axis = figure.add_subplot(grid[1, 2])
    note_axis.axis("off")
    note_axis.text(0, 1.15, "Grey: unsupported triangles\nPurple: evaluated blackout\nDots: subsampled map locations", fontsize=6.5, va="top", linespacing=1.35)

    x = np.arange(len(PATIENT_IDS))
    for method, marker in (("screened", "o"), ("passive", "s"), ("graph", "^")):
        chosen = summary[summary["method"] == method].set_index("patient_id").loc[list(PATIENT_IDS)]
        bottom[0].plot(x, chosen["rmse_area_weighted"], marker=marker, markersize=3.1,
                       linewidth=.85, color=METHOD_COLOURS[method], label=method,
                       markerfacecolor="none" if method != "screened" else METHOD_COLOURS[method])
    bottom[0].set(title="d  Held-out perivenous score", ylabel="Area-weighted RMSE")
    bottom[0].set_xticks(x, PATIENT_IDS)
    bottom[0].legend(frameon=False)
    bottom[0].grid(axis="y", alpha=.2)

    for method, marker in (("passive", "o"), ("graph", "s")):
        chosen = capacity[capacity["method"] == method]
        bottom[1].scatter(chosen["reference_normalised_capacity"], chosen["normalised_capacity"],
                          s=14, marker=marker, alpha=.65, color=METHOD_COLOURS[method], label=method,
                          edgecolors="none")
    limit = 1.04 * float(max(capacity["reference_normalised_capacity"].max(), capacity["normalised_capacity"].max()))
    bottom[1].plot([0, limit], [0, limit], color="black", lw=.7, ls="--")
    bottom[1].set(xlim=(0, limit), ylim=(0, limit), title="e  Hidden-PV capacity",
                  xlabel="Full-map reference", ylabel="Reconstructed")
    bottom[1].set_aspect("equal", adjustable="box")
    bottom[1].legend(frameon=False, loc="lower right")
    bottom[1].grid(alpha=.2)

    rmse = summary.pivot(index="patient_id", columns="method", values="rmse_area_weighted").loc[list(PATIENT_IDS)]
    for index, comparator in enumerate(("screened", "passive")):
        differences = (rmse["graph"] - rmse[comparator]).to_numpy(dtype=float)
        bottom[2].scatter(index + np.linspace(-.07, .07, len(differences)), differences,
                          s=16, color="#444444", alpha=.85, edgecolors="none")
        bottom[2].plot([index - .19, index + .19], [differences.mean()] * 2,
                       color=METHOD_COLOURS["graph"], lw=2.0)
    bottom[2].axhline(0, color="black", lw=.7, ls="--")
    bottom[2].set_xticks([0, 1], ["Graph −\nscreened", "Graph −\npassive"])
    bottom[2].set(title="f  Paired patient contrasts", ylabel="Patient RMSE difference", xlim=(-.4, 1.4))
    bottom[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    bottom[2].grid(axis="y", alpha=.2)
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)


def _preview_geometry(cohort_root: Path, output_base: Path) -> None:
    # This preview uses source anatomy and observations only.  It deliberately
    # neither invents reconstruction fields nor creates a result figure.
    import run_zenodo_pvi_reconstruction as runner
    from scipy.sparse.csgraph import dijkstra

    _style()
    mesh, boundaries, _, _ = runner._load_patient(cohort_root, "P3")
    score, signed = runner._threshold_anchored_score(mesh.point_data["bi"])
    nodes = np.unique(np.concatenate([boundaries[label] for label in ("LSPV", "LIPV")]))
    distance = dijkstra(runner._edge_graph(mesh), directed=False, indices=nodes, min_only=True)
    evaluation = distance <= runner.EVALUATION_WIDTH_MM
    blackout = distance <= runner.BLACKOUT_WIDTH_MM
    contacts, confidence, forcing = runner._observation_fields(mesh, signed, blackout)
    observed = np.zeros_like(confidence)
    support = confidence > 0
    observed[support] = .5 * (1 + forcing[support] / confidence[support])
    view = anatomical_view(mesh.points, mesh.triangles, boundaries)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    first = surface_panel(axes[0], view, score, title="P3 clinical-map score", landmarks=boundaries)
    surface_panel(axes[1], view, observed, title="Sparse map observations", support=support,
                  evaluation=evaluation, contacts=contacts)
    fig.colorbar(first, ax=axes, fraction=.035, label="Threshold-anchored barrier score")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-anatomy", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tmp/patient_anatomy_preview"))
    args = parser.parse_args()
    if args.preview_anatomy is None:
        parser.error("use --preview-anatomy COHORT_ROOT or import make_patient_surface_figure")
    _preview_geometry(args.preview_anatomy, args.output)
