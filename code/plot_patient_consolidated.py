"""Make the compact patient figures from recorded reconstruction outputs.

No reconstruction is performed and no numerical field is modified. The main
figure pairs reference and reconstructed capacities on the identical 10-mm
PV annulus. Markers are deliberately unfilled or crossed to show methods
whose numerical coordinates nearly coincide; no coordinate jitter is used.
"""
from __future__ import annotations

import json
import hashlib
import io
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from patient_surface_plot import (
    PATIENT_IDS, METHOD_COLOURS, anatomical_view, surface_panel,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXTENSION = DATA / "patient_extension"
FIGURES = ROOT / "figures"
WIDTH_IN = 12.2 / 2.54
METHODS = ("screened", "passive", "graph")
MARKERS = {"screened": "s", "passive": "o", "graph": "x"}
LINESTYLES = {"screened": "-", "passive": "--", "graph": ":"}


def style():
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix",
        "font.size": 7.1, "axes.titlesize": 7.6, "axes.labelsize": 7.1,
        "xtick.labelsize": 6.7, "ytick.labelsize": 6.7,
        "legend.fontsize": 7.0, "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 450, "figure.dpi": 160,
    })


def method_handles(*, lines=True):
    return [Line2D([], [], color=METHOD_COLOURS[m], marker=MARKERS[m],
                   markerfacecolor="none", markeredgewidth=.85,
                   markersize={"screened": 5.6, "passive": 4.5, "graph": 3.6}[m],
                   linewidth=.9, linestyle=LINESTYLES[m] if lines else "none",
                   label=m.capitalize()) for m in METHODS]


def save(figure, basename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    # A fixed canvas preserves the intended 12.2-cm width. Text remains vector;
    # the anatomical surface collections are rasterized at 450 dpi by their
    # established plotting helper, while contours and markers remain vector.
    for suffix in ("pdf", "png"):
        buffer = io.BytesIO()
        figure.savefig(buffer, format=suffix, dpi=450)
        contents = buffer.getvalue()
        if suffix == "pdf":
            assert contents.rstrip().endswith(b"%%EOF"), "incomplete PDF export"
        destination = FIGURES / f"{basename}.{suffix}"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(contents)
        temporary.replace(destination)
    plt.close(figure)


def load_records():
    records = {}
    for width in (3, 10):
        path = EXTENSION / f"P3_left_pair_w{width}_h0_dt0p01.npz"
        with np.load(path) as values:
            records[width] = {key: values[key] for key in values.files}
    for field in ("points", "triangles", "reference_score"):
        np.testing.assert_array_equal(records[3][field], records[10][field])
    return records


def main_figure(records, summary, capacity, detailed_capacity):
    fig = plt.figure(figsize=(WIDTH_IN, 5.85))
    axes = [fig.add_axes(rect) for rect in (
        [.015, .730, .475, .240], [.510, .730, .475, .240],
        [.015, .445, .475, .240], [.510, .445, .475, .240],
    )]
    original = records[10]
    boundaries = {label: original[f"boundary_{label}"]
                  for label in ("LSPV", "LIPV", "RSPV", "RIPV")}
    view = anatomical_view(original["points"], original["triangles"], boundaries)
    first = surface_panel(axes[0], view, original["reference_score"],
                          title="(a) P3 reference score", landmarks=boundaries)
    support = original["confidence"] > 0
    observed = np.zeros_like(original["confidence"])
    observed[support] = .5 * (1 + original["forcing"][support] / original["confidence"][support])
    surface_panel(axes[1], view, observed, title="(b) Observations, 10 mm",
                  support=support, evaluation=original["evaluation_mask"],
                  contacts=original["contact_nodes"])
    for axis, width, letter in ((axes[2], 3, "c"), (axes[3], 10, "d")):
        rec = records[width]
        values = np.clip(.5 * (1 + rec["graph_state"]), 0, 1)
        surface_panel(axis, view, values, title=f"({letter}) Graph, {width}-mm band",
                      evaluation=rec["evaluation_mask"])
    for axis in axes:
        for item in axis.texts:
            # Retain readable anatomical landmarks and scale bars at the
            # actual journal width; this does not change camera or geometry.
            item.set_fontsize(6.4)
    coloraxis = fig.add_axes([.30, .419, .40, .009])
    bar = fig.colorbar(first, cax=coloraxis, orientation="horizontal", ticks=[0, .5, 1])
    bar.set_label("Voltage-derived barrier score", fontsize=7.0, labelpad=2)
    bar.ax.xaxis.set_label_position("top")
    bar.ax.tick_params(labelsize=6.6, pad=1.0, length=2)

    quantitative = [fig.add_axes(rect) for rect in (
        [.115, .105, .213, .235], [.440, .105, .213, .235], [.765, .105, .213, .235],
    )]
    for axis, frame, endpoint, title, ylabel in (
        (quantitative[0], summary, "rmse_area_weighted", "(e) Score RMSE", "Common-region RMSE"),
        (quantitative[1], capacity, "capacity_absolute_error", "(f) Capacity error", "Mean absolute error"),
    ):
        for method in METHODS:
            chosen = frame[frame.method == method]
            means = chosen.groupby("evaluation_width_mm")[endpoint].mean()
            axis.plot(means.index, means.values, color=METHOD_COLOURS[method],
                      marker=MARKERS[method], markerfacecolor="none",
                      markeredgewidth=.8, linestyle=LINESTYLES[method],
                      markersize={"screened": 5.6, "passive": 4.5, "graph": 3.5}[method],
                      linewidth=.85, zorder=2 + METHODS.index(method))
        axis.set_title(title, loc="left", pad=5)
        axis.set_xlabel("Hidden width (mm)", labelpad=3)
        axis.set_ylabel(ylabel, labelpad=3)
        axis.set_xticks([3, 5, 10])
        axis.set_xlim(2.5, 10.5)
        axis.grid(axis="y", alpha=.17, linewidth=.5)
    quantitative[0].set_ylim(.175, .285)
    quantitative[0].set_yticks([.18, .21, .24, .27])
    quantitative[1].set_ylim(.14, .38)
    quantitative[1].set_yticks([.15, .20, .25, .30, .35])

    axis = quantitative[2]
    axis.plot([0, 1], [0, 1], color="#222222", linestyle="--", linewidth=.7, zorder=1)
    # Draw all 24 PV values per estimator at their measured coordinates.
    # Large open passive circles preserve their visibility beneath graph x's.
    for method in METHODS:
        chosen = detailed_capacity[detailed_capacity.method == method]
        size = {"screened": 30, "passive": 20, "graph": 10}[method]
        extra = {"facecolors": "none", "edgecolors": METHOD_COLOURS[method]} \
                if method != "graph" else {"color": METHOD_COLOURS[method]}
        axis.scatter(chosen.reference_normalised_capacity, chosen.normalised_capacity,
                     marker=MARKERS[method], s=size, linewidths=.8, alpha=.95,
                     zorder=2 + METHODS.index(method), **extra)
    axis.set_title("(g) PV capacity\n10-mm band", loc="left", pad=5)
    axis.set_xlabel("Reference", labelpad=3)
    axis.set_ylabel("Reconstructed", labelpad=3)
    axis.set_xlim(-.04, 1.02); axis.set_ylim(-.04, 1.02)
    axis.set_xticks([0, .5, 1]); axis.set_yticks([0, .5, 1])
    axis.set_aspect("equal", adjustable="box")
    # Equal axis scales are necessary to interpret the identity comparison.
    axis.grid(alpha=.17, linewidth=.5)
    handles = method_handles()
    handles.append(Line2D([], [], color="#222222", linestyle="--", linewidth=.7,
                          label="Identity (g)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .010),
               ncol=4, frameon=False, columnspacing=.9, handlelength=1.8,
               handletextpad=.4, borderaxespad=0)
    save(fig, "fig_patient_consolidated")


def secondary_figure():
    summary = pd.read_csv(DATA / "zenodo_pvi_patient_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.25))
    fig.subplots_adjust(left=.11, right=.985, bottom=.24, top=.79, wspace=.52)
    x = np.arange(len(PATIENT_IDS))
    for method in METHODS:
        chosen = summary[summary.method == method].set_index("patient_id").loc[list(PATIENT_IDS)]
        axes[0].plot(x, chosen.rmse_area_weighted, marker=MARKERS[method],
                     markersize=4.6 if method == "passive" else 3.7,
                     markerfacecolor="none", markeredgewidth=.85,
                     linestyle=LINESTYLES[method], linewidth=.85,
                     color=METHOD_COLOURS[method], zorder=2 + METHODS.index(method))
    axes[0].set_title("(a) Patient RMSE, 10-mm band", loc="left")
    axes[0].set_ylabel("Area-weighted RMSE")
    axes[0].set_xticks(x, PATIENT_IDS)
    axes[0].grid(axis="y", alpha=.18, linewidth=.5)
    pivot = summary.pivot(index="patient_id", columns="method", values="rmse_area_weighted").loc[list(PATIENT_IDS)]
    for index, comparator in enumerate(("screened", "passive")):
        delta = (pivot.graph - pivot[comparator]).to_numpy()
        # Fixed horizontal offsets distinguish the six categorical observations;
        # the measured RMSE contrasts are drawn without vertical displacement.
        axes[1].scatter(index + np.linspace(-.12, .12, len(delta)), delta,
                        s=13, c="#444444", edgecolors="none", zorder=3)
        axes[1].plot([index-.20, index+.20], [delta.mean()]*2,
                     color=METHOD_COLOURS["graph"], linewidth=1.7, zorder=4)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=.65)
    axes[1].set_title("(b) Paired patient contrasts", loc="left")
    axes[1].set_ylabel("Patient RMSE difference")
    axes[1].set_xticks([0, 1], ["Graph −\nscreened", "Graph −\npassive"])
    axes[1].set_xlim(-.4, 1.4)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    axes[1].grid(axis="y", alpha=.18, linewidth=.5)
    fig.legend(handles=method_handles(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .005), handlelength=2.0, columnspacing=1.5)
    save(fig, "fig_patient_secondary")



def surface_comparison_figure(records):
    fig = plt.figure(figsize=(WIDTH_IN, 2.95))
    grid = fig.add_gridspec(2, 4, left=.065, right=.985, top=.93, bottom=.19,
                           hspace=.11, wspace=.10)
    base = records[3]
    boundaries = {label: base[f"boundary_{label}"]
                  for label in ("LSPV", "LIPV", "RSPV", "RIPV")}
    view = anatomical_view(base["points"], base["triangles"], boundaries)
    for row, width in enumerate((3, 10)):
        rec = records[width]
        for column, method in enumerate(("reference", "screened", "passive", "graph")):
            axis = fig.add_subplot(grid[row, column])
            values = rec["reference_score"] if method == "reference" else np.clip(.5*(1+rec[method+"_state"]), 0, 1)
            mappable = surface_panel(axis, view, values,
                title=f"({chr(ord('a')+4*row+column)}) {method.capitalize()}",
                evaluation=rec["evaluation_mask"])
            axis.title.set_fontsize(6.7)
            for text in axis.texts:
                text.set_fontsize(5.5)
            if column == 0:
                axis.text(-.07, .5, f"{width}-mm band", fontsize=6.3,
                          rotation=90, va="center", ha="right", transform=axis.transAxes)
    coloraxis = fig.add_axes([.30, .10, .40, .016])
    bar = fig.colorbar(mappable, cax=coloraxis, orientation="horizontal", ticks=[0, .5, 1])
    bar.set_label("Voltage-derived barrier score", fontsize=6.6, labelpad=1.5)
    bar.ax.xaxis.set_label_position("top")
    bar.ax.tick_params(labelsize=6.2, pad=1, length=2)
    save(fig, "fig_patient_surface_comparisons")


def main():
    style()
    records = load_records()
    summary = pd.read_csv(EXTENSION / "patient_extension_patient_summary.csv")
    summary = summary[(summary.mesh_level == 0) & np.isclose(summary.pseudo_dt, .01)
                      & (summary.metric_support == "common_3mm")]
    capacity = pd.read_csv(EXTENSION / "patient_extension_patient_capacity.csv")
    capacity = capacity[(capacity.mesh_level == 0) & np.isclose(capacity.pseudo_dt, .01)]
    detailed = pd.read_csv(DATA / "zenodo_pvi_capacity.csv")
    assert len(summary) == 54 and len(capacity) == 54 and len(detailed) == 72
    assert set(summary.patient_id) == set(PATIENT_IDS)
    assert set(summary.evaluation_width_mm) == {3, 5, 10}
    assert np.all(detailed.annulus_width_mm == 10)
    keys = ["patient_id", "mask", "pv_label"]
    references = detailed.groupby(keys).reference_normalised_capacity
    assert references.nunique().eq(1).all() and len(references) == 24
    # This plotted severe condition is the same endpoint as the original run.
    extension_detail = pd.read_csv(EXTENSION / "patient_extension_capacity.csv")
    extension_detail = extension_detail[(extension_detail.mesh_level == 0)
        & np.isclose(extension_detail.pseudo_dt, .01)
        & np.isclose(extension_detail.evaluation_width_mm, 10)]
    paired = detailed.merge(extension_detail, on=keys+["method"], suffixes=("_original", "_extension"),
                            validate="one_to_one")
    assert len(paired) == 72
    for endpoint in ("normalised_capacity", "reference_normalised_capacity"):
        np.testing.assert_allclose(paired[endpoint+"_original"], paired[endpoint+"_extension"],
                                   rtol=1e-13, atol=1e-14)
    main_figure(records, summary, capacity, detailed)
    secondary_figure()
    surface_comparison_figure(records)
    metadata = {
        "figure_width_cm": 12.2, "main_figure_height_cm": 5.85*2.54,
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                EXTENSION / "P3_left_pair_w3_h0_dt0p01.npz",
                EXTENSION / "P3_left_pair_w10_h0_dt0p01.npz",
                EXTENSION / "patient_extension_patient_summary.csv",
                EXTENSION / "patient_extension_patient_capacity.csv",
                EXTENSION / "patient_extension_capacity.csv",
                DATA / "zenodo_pvi_capacity.csv", DATA / "zenodo_pvi_patient_summary.csv",
            ]},
        "methods": list(METHODS), "patient_ids": list(PATIENT_IDS),
        "surface_case": "P3, left-pair withholding, original mesh, pseudo-time step 0.01",
        "surface_records": [str(EXTENSION.relative_to(ROOT) / f"P3_left_pair_w{w}_h0_dt0p01.npz") for w in (3, 10)],
        "score_curve_domain": "identical original-node 0--3-mm evaluated band at all three missingness widths",
        "score_curve_aggregation": "area-weighted RMSE per mask, two masks averaged per patient, then mean of six patients",
        "capacity_curve_domain": "fixed original 10-mm PV annuli at all missingness widths",
        "capacity_curve_aggregation": "absolute errors averaged across four PVs within each patient, then mean of six patients",
        "capacity_comparison": "24 PV annuli per estimator; reference and reconstructed field evaluated on the same annulus and normalized by the same coefficient-one capacity",
        "capacity_comparison_coordinate_jitter": False,
        "capacity_original_extension_max_difference": float(np.max(np.abs(paired.normalised_capacity_original-paired.normalised_capacity_extension))),
        "reference_capacity_mean": float(detailed[detailed.method == "graph"].reference_normalised_capacity.mean()),
        "graph_capacity_mean": float(detailed[detailed.method == "graph"].normalised_capacity.mean()),
        "capacity_overestimation_all_three_methods_all_hidden_pvs": bool(np.all(detailed.normalised_capacity > detailed.reference_normalised_capacity)),
        "saturated_score_for_display_only": True,
    }
    (DATA / "patient_consolidated_figure.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
