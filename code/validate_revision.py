"""Independently check the additional studies against their declared designs.

The checks recompute selected readouts from stored fields, verify paired support
and source hashes, and reject incomplete factorial designs. They do not require a
preferred scientific outcome. Run after the new experiments and summaries finish.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
METHODS = ("screened", "passive", "graph")
FOUR_METHODS = ("reference",) + METHODS
GEOMETRIES = ("complete_ring", "narrow_gap", "wide_gap", "two_gaps", "oblique_gap")
EP_CASES = (("complete_ring", 0), ("narrow_gap", 0), ("wide_gap", 90))
PATIENTS = ("P1", "P3", "P4", "P5", "P6", "P7")
COUNTS: dict[str, int] = {}
NOTES: list[str] = []
CURRENT = ""


def require(condition, message):
    COUNTS[CURRENT] = COUNTS.get(CURRENT, 0) + 1
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, message, atol=2.0e-11, rtol=2.0e-10):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True), message)


def csv(name):
    path = DATA / name
    require(path.is_file(), f"missing {name}")
    return pd.read_csv(path)


def unique(frame, columns, message):
    require(not frame.duplicated(columns).any(), message)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def sector_mask(n, width, angle):
    x = np.arange(n) * 60.0 / n - 30.0
    X, Y = np.meshgrid(x, x, indexing="ij")
    radial = np.hypot(X, Y)
    arc = 15.0 * np.abs(np.angle(np.exp(1j * (np.arctan2(Y, X) - np.deg2rad(angle)))))
    return (np.abs(radial - 15.0) <= 3.0) & (arc <= width / 2.0 + 1.0)


def synthetic():
    from run_applied import GEOMETRY_SPECS, geometry_score_field
    specifications = {s["geometry"]: s["gaps"] for s in GEOMETRY_SPECS}
    coordinates = np.arange(241) * 60.0 / 241
    X, Y = np.meshgrid(coordinates, coordinates, indexing="ij")
    regimes = (
        ("support_exclusion", DATA, DATA / "reconstruction_extension_fields", (6,)),
        ("boundary_retained", DATA / "reconstruction_boundary_control", DATA / "reconstruction_boundary_fields", (2, 6)),
    )
    expected_main = set(itertools.product(GEOMETRIES, (2, 6, 12), (0, 90, 180, 270), METHODS))
    for regime, directory, fields_directory, refinement_widths in regimes:
        frame = pd.read_csv(directory / "reconstruction_extension_blocks.csv")
        keys = ["geometry", "blackout_arc_width_mm", "blackout_angle_deg", "reconstruction_n", "pseudo_dt", "method"]
        unique(frame, keys, f"{regime}: duplicate result rows")
        main = frame[(frame.reconstruction_n == 81) & np.isclose(frame.pseudo_dt, .01)]
        require(set(map(tuple, main[keys[:3] + ["method"]].to_numpy())) == expected_main,
                f"{regime}: incomplete or unexpected main factorial design")
        expected_refinement = {(g, w, a, n, dt, method)
            for (g, a), w, (n, dt), method in itertools.product(
                EP_CASES, refinement_widths, ((121, .01), (161, .01), (81, .005)), METHODS)}
        ref = frame.drop(main.index)
        require(set(map(tuple, ref[keys].to_numpy())) == expected_refinement,
                f"{regime}: incomplete mesh/time controls")
        close(frame.horizon, 1.2, "synthetic comparisons change pseudo-time horizon")
        close(frame.terminal_window_time, .12, "synthetic terminal window changes duration")
        require((frame.evaluation_grid_n == 241).all(), "synthetic readouts use unequal grids")
        require((frame.holdout_confidence_max == 0).all() and (frame.holdout_forcing_max_abs == 0).all(),
                "synthetic hidden regions contain observation forcing")
        seed = main.blackout_angle_deg.map({0: 3101, 90: 3103, 180: 3107, 270: 3109})
        require(np.array_equal(main.acquisition_seed, seed), "acquisition seed differs from declared rotation map")
        observations = {}
        cohorts = {}
        saved_rows = []
        diagnostic_rows = []
        primary_keys = set(map(tuple, frame[keys].to_numpy()))
        for path in sorted(fields_directory.glob("*.npz")):
            with np.load(path, allow_pickle=False) as f:
                meta = json.loads(str(f["metadata_json"]))
                rows = json.loads(str(f["rows_json"]))
                for row in rows:
                    (saved_rows if tuple(row[k] for k in keys) in primary_keys else diagnostic_rows).append(row)
                n = int(meta["reconstruction_n"])
                geometry, width, angle = meta["geometry"], meta["blackout_arc_width_mm"], meta["blackout_angle_deg"]
                hidden = f["evaluation_mask"].astype(bool)
                require(np.array_equal(hidden, sector_mask(n, width, angle)), "stored synthetic mask disagrees with physical sector")
                require(np.all(f["confidence"][hidden] == 0) and np.all(f["data_forcing"][hidden] == 0),
                        f"{path.name}: hidden data leakage")
                observation_bound = max(1., float(np.max(np.abs(f["contact_score"]))))
                require(np.all(f["confidence"] >= 0) and np.all(np.abs(f["data_forcing"]) <= observation_bound * f["confidence"] + 1e-12),
                        "synthetic forcing exceeds finite observation-weight bound")
                require(np.all(f["data_forcing"][f["confidence"] == 0] == 0), "forcing where synthetic confidence vanishes")
                require(all(np.isfinite(f[m]).all() for m in ("truth",) + METHODS), "non-finite synthetic field")
                obs = tuple(f[k].copy() for k in ("contact_indices", "contact_x", "contact_y", "contact_score"))
                key = (geometry, width, angle)
                if key in observations:
                    require(all(np.array_equal(a, b) for a, b in zip(obs, observations[key])), "refinement changes observations")
                observations[key] = obs
                cohorts.setdefault((width, angle), []).append((meta["coordinate_sha256"], meta["noise_sha256"]))
                # The common-core count must match the same physical support at every width and grid.
                common_count = int(sector_mask(241, 2, angle).sum())
                common_mask = sector_mask(241, 2, angle)
                analytic_truth = geometry_score_field(X, Y, specifications[geometry])
                for row in rows:
                    require(row["common_core_n_eval"] == common_count, "synthetic width comparison changes common support")
                    close(row["capacity_abs_error"], abs(row["normalized_capacity"] - row["truth_normalized_capacity"]), "capacity error arithmetic")
                    close(row["gap_count_abs_error"], abs(row["gap_count"] - row["truth_gap_count"]), "gap error arithmetic")
                    interpolated = resample(resample(f[row["method"]], 241, axis=0), 241, axis=1).real
                    errors = interpolated[common_mask] - analytic_truth[common_mask]
                    close(row["common_core_score_rmse"], np.sqrt(np.mean(errors**2)), "synthetic common-core RMSE disagrees with saved field")
                    close(row["common_core_score_mae"], np.mean(np.abs(errors)), "synthetic common-core MAE disagrees with saved field")
        stored = pd.DataFrame(saved_rows).sort_values(keys).reset_index(drop=True)
        table = frame.sort_values(keys).reset_index(drop=True)
        require(len(stored) == len(table), f"{regime}: CSV omits stored completed cases")
        require(stored[keys].equals(table[keys]), f"{regime}: table/field case mismatch")
        for endpoint in ("common_core_score_rmse", "masked_score_rmse", "normalized_capacity", "gap_count", "total_cpu_seconds"):
            close(stored[endpoint], table[endpoint], f"{regime}: {endpoint} disagrees with checkpoint")
        for values in cohorts.values():
            require(len(set(values)) == 1, "contact-coordinate or noise pool changes across geometries/refinements")
        for geometry, angle in itertools.product(GEOMETRIES, (0, 90, 180, 270)):
            for small, large in ((2, 6), (6, 12)):
                lo, hi = observations[(geometry, small, angle)], observations[(geometry, large, angle)]
                require(set(hi[0]).issubset(lo[0]), "larger blackout adds previously unavailable contacts")
                positions = {int(v): i for i, v in enumerate(lo[0])}
                inds = np.array([positions[int(v)] for v in hi[0]])
                require(all(np.array_equal(a[inds], b) for a, b in zip(lo[1:], hi[1:])), "nested contact value or noise changed")
        summary = pd.read_csv(directory / "reconstruction_extension_summary.csv")
        metrics = ["common_core_score_rmse", "masked_score_rmse", "common_core_dice", "capacity_abs_error", "gap_count_abs_error"]
        independent = main.groupby(["geometry", "blackout_arc_width_mm", "method"])[metrics].mean().reset_index()
        independent = independent.groupby(["blackout_arc_width_mm", "method"])[metrics].mean().sort_index()
        actual = summary.set_index(["blackout_arc_width_mm", "method"]).sort_index()
        close(actual[metrics], independent[metrics], "synthetic summary changes the geometry analysis unit")
        if regime == "boundary_retained":
            diagnostic = pd.DataFrame(diagnostic_rows)
            expected_diagnostic = {(g, 6, 270, n, dt, method)
                for g, (n, dt), method in itertools.product(
                    ("complete_ring", "narrow_gap", "wide_gap", "two_gaps"),
                    ((121, .01), (161, .01), (81, .005)), METHODS)}
            require(set(map(tuple, diagnostic[keys].to_numpy())) == expected_diagnostic,
                    "result-triggered topology control incomplete or mixed into primary study")
            reported = pd.read_csv(directory / "topology_diagnostic_blocks.csv").sort_values(keys)
            close(reported[metrics], diagnostic.sort_values(keys)[metrics], "topology diagnostic table/fields mismatch")
    NOTES.append("Synthetic acquisition controls are separate designs: boundary retention was added after observing the original stress-test failure, not prespecified before all outcomes.")
    NOTES.append("Synthetic contact scores use untruncated Gaussian noise and may exceed [-1,1]; the applicable forcing bound is |f| <= C_obs*lambda with C_obs=max(1,max|observed score|).")


def transfer_score(score, n):
    """Re-evaluate the declared bilinear/subcell transfer, independently of the EP driver."""
    out = np.zeros((n, n))
    for ox, oy in itertools.product((1 / 6, 1 / 2, 5 / 6), repeat=2):
        x = (np.arange(n) + ox) * score.shape[0] / n
        y = (np.arange(n) + oy) * score.shape[1] / n
        X, Y = np.meshgrid(x, y, indexing="ij")
        s = map_coordinates(score, np.array((X, Y)), order=1, mode="grid-wrap", prefilter=False)
        out += .001 + .999 / (1 + np.exp(np.clip(8 * s, -60, 60)))
    return out / 9


def ep_target(n, angle, direction):
    x = (np.arange(n) + .5) * 60 / n - 30
    X, Y = np.meshgrid(x, x, indexing="ij")
    radius = np.hypot(X, Y)
    theta = np.abs(np.angle(np.exp(1j * (np.arctan2(Y, X) - np.deg2rad(angle)))))
    lo, hi = (20, 23) if direction == "exit" else (7, 10)
    return (radius >= lo) & (radius <= hi) & (theta <= np.pi / 12)


def reconstruction_ep():
    frame = csv("reconstruction_ep_runs.csv")
    configurations = {
        "production": ((2, 6, 12), 81, .01, 121, .02),
        "boundary_production": ((2, 6, 12), 81, .01, 121, .02),
        "ep_spatial": ((6,), 81, .01, 161, .02),
        "ep_temporal": ((6,), 81, .01, 121, .01),
        "reconstruction_spatial": ((6,), 121, .01, 121, .02),
        "reconstruction_temporal": ((6,), 81, .005, 121, .02),
        "boundary_ep_spatial": ((2, 6), 81, .01, 161, .02),
        "boundary_ep_temporal": ((2, 6), 81, .01, 121, .01),
    }
    keys = ["study", "geometry", "blackout_arc_width_mm", "angle_deg", "method", "direction"]
    unique(frame, keys, "duplicate reconstruction-to-EP cases")
    expected = set()
    for study, (widths, rn, rdt, en, edt) in configurations.items():
        for (geometry, angle), width, method, direction in itertools.product(EP_CASES, widths, FOUR_METHODS, ("entrance", "exit")):
            expected.add((study, geometry, width, angle, method, direction))
        part = frame[frame.study == study]
        for column, value in (("reconstruction_n", rn), ("reconstruction_dt", rdt), ("ep_n", en), ("ep_dt_ms", edt)):
            close(part[column], value, f"{study}: unplanned simultaneous change in {column}")
    require(set(map(tuple, frame[keys].to_numpy())) == expected, "missing/unexpected reconstruction-to-EP design cells")
    for column, value in (("t_end_ms", 210), ("stimulus_amplitude", 1.2), ("stimulus_duration_ms", 2), ("stimulus_radius_mm", 3.5), ("capture_fraction_threshold", .8)):
        close(frame[column], value, f"EP protocol differs in {column}")
    require((frame.face_average == "harmonic").all(), "EP face averaging differs between methods")
    require((frame.cfl < .95).all(), "EP diffusion CFL violated")
    require(frame.max_diffusion_mass_defect.max() < 1e-10, "EP flux is not conservative")
    hashes, fields, activations, coefficients = {}, {}, {}, {}
    for row in frame.itertuples():
        path = ROOT / row.source_file
        if path not in hashes:
            hashes[path] = file_hash(path)
        require(hashes[path] == row.source_sha256, "EP source changed since solve")
        if path not in fields:
            with np.load(path, allow_pickle=False) as f:
                fields[path] = {m: f["truth" if m == "reference" else m] for m in FOUR_METHODS}
        score = fields[path][row.method]
        require(array_hash(score) == row.field_sha256, "EP method refers to wrong source field")
        checkpoint = ROOT / row.checkpoint
        if checkpoint not in activations:
            with np.load(checkpoint, allow_pickle=False) as f:
                activations[checkpoint] = f["activation"]
                coefficients[checkpoint] = f["diffusivity"]
        activation = activations[checkpoint]
        require(array_hash(activation) == row.activation_field_sha256, "stored activation hash changed")
        close(coefficients[checkpoint], transfer_score(score, row.ep_n), "stored EP coefficient is not common declared transfer", atol=3e-13, rtol=3e-13)
        target = ep_target(row.ep_n, row.angle_deg, row.direction)
        require(target.sum() == row.target_cell_count, "EP target geometry differs from protocol")
        fraction = np.mean(np.isfinite(activation[target]))
        close(fraction, row.target_activated_fraction, "EP target fraction does not match activation field")
        close(np.mean(np.isfinite(activation)), row.activated_fraction, "EP whole-sheet fraction differs from field")
        require(int(fraction >= .8) == row.captured_by_horizon, "EP capture label differs from declared threshold")
        require(np.isnan(row.crossing_time_ms) == (not row.captured_by_horizon), "noncapture assigned an arrival time")
        first_arrival = np.nanmin(activation[target]) if row.captured_by_horizon else np.nan
        close(row.crossing_time_ms, first_arrival, "reported conditional EP arrival is not the first target activation")
    comparison = csv("reconstruction_ep_comparisons.csv")
    require(len(comparison) == len(frame) * 3 // 4, "reference rows retained/omitted incorrectly in EP contrasts")
    index = frame.set_index(keys)
    for row in comparison.itertuples():
        key = tuple(getattr(row, k) for k in keys)
        refkey = key[:4] + ("reference", key[-1])
        ref = index.loc[refkey]
        both = bool(row.captured_by_horizon and ref.captured_by_horizon)
        require(row.both_capture == both, "conditional arrival support differs from paired capture")
        require(row.capture_disagreement == int(row.captured_by_horizon != ref.captured_by_horizon), "capture discrepancy arithmetic")
        close(row.target_activated_fraction_error, row.target_activated_fraction - ref.target_activated_fraction, "target error paired to wrong reference")
        close(row.conditional_arrival_error_ms, row.crossing_time_ms - ref.crossing_time_ms if both else np.nan, "arrival error imputes noncapture or uses wrong reference")
    refinement = csv("reconstruction_ep_refinement.csv")
    require(len(refinement) == 192, "EP refinement table incomplete")
    for row in refinement.itertuples():
        production = "boundary_production" if row.acquisition == "boundary_coverage" else "production"
        baseline = index.loc[(production, row.geometry, row.blackout_arc_width_mm, row.angle_deg, row.method, row.direction)]
        require(row.capture_changed_from_production == int(row.captured_by_horizon != baseline.captured_by_horizon),
                "EP refinement capture-change flag paired to wrong production state")
        close(row.target_fraction_change_from_production, row.target_activated_fraction - baseline.target_activated_fraction,
              "EP refinement fraction change arithmetic")
    topology = csv("reconstruction_ep_topology_capacity.csv")
    require(len(topology) == 72, "EP-grid topology/capacity analysis incomplete")
    require(topology.normalized_capacity.between(0, 1, inclusive="neither").all(), "EP capacity has impossible normalized range")
    for column in ("capacity_energy", "capacity_inner_flux", "capacity_outer_flux"):
        close(topology[column], topology.capacity, "EP capacity flux/energy identity", atol=1e-10, rtol=1e-9)
    ratios = topology.assign(normalizer=topology.capacity / topology.normalized_capacity).groupby("ep_n").normalizer
    require((ratios.max() - ratios.min()).max() < 1e-10, "EP capacity normalization changes electrode domain across fields")
    for row in topology.itertuples():
        original = index.loc[tuple(getattr(row, k) for k in keys)]
        require(row.checkpoint == original.checkpoint and row.field_sha256 == original.field_sha256,
                "topology/capacity analysis uses different source than EP")
    NOTES.append("Reconstruction-to-EP validation verifies coefficient transfer, source hashes, capture, and conditional arrival rules; it imposes no preferred capture outcome.")


def patient_case_key(patient, mask, width, level, dt):
    return f"{patient}_{mask}_w{width}_h{level}_dt{str(dt).replace('.', 'p')}"


def patient_reconstruction():
    directory = DATA / "patient_extension"
    frame = csv("patient_extension/patient_extension_metrics.csv")
    cap = csv("patient_extension/patient_extension_capacity.csv")
    cases = {(p, mask, width, 0, .01) for p, mask, width in itertools.product(PATIENTS, ("left_pair", "right_pair"), (3, 5, 10))}
    cases |= {(p, mask, 10, 0, .005) for p, mask in itertools.product(PATIENTS, ("left_pair", "right_pair"))}
    cases |= {("P3", "left_pair", 10, 1, dt) for dt in (.01, .005)}
    keys = ["patient_id", "mask", "evaluation_width_mm", "mesh_level", "pseudo_dt"]
    require(set(map(tuple, frame[keys].to_numpy())) == cases, "incomplete patient missingness or discretisation cases")
    require(len(frame) == 300 and len(cap) == 300, "patient tables have incorrect case/support counts")
    unique(frame, keys + ["method", "metric_support"], "duplicate patient score rows")
    unique(cap, keys + ["method", "pv_label"], "duplicate patient capacity rows")
    close(frame.pseudo_horizon, 1.2, "patient pseudo-time horizon changed")
    close(frame.terminal_duration, .12, "patient terminal window changed")
    close(frame.terminal_states * frame.pseudo_dt, .12, "patient time refinement changes averaging window")
    require((frame.blackout_confidence_max == 0).all() and (frame.blackout_forcing_max_abs == 0).all(), "patient hidden forcing present")
    require(frame.input_hidden_value_poison_check.all(), "patient hidden-value poisoning check failed")
    base = {}
    for patient, mask, width, level, dt in sorted(cases):
        key = patient_case_key(patient, mask, width, level, dt)
        with np.load(directory / f"{key}.npz", allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        support = data["evaluation_mask"].astype(bool)
        common = data["common_evaluation_mask"].astype(bool)
        blackout = data["blackout_mask"].astype(bool)
        area = data["node_area"]
        require(np.all(area > 0) and np.isfinite(area).all(), "patient quadrature is not positive")
        require(np.all(~common | support) and np.all(~support | blackout), "patient nested common/evaluation/blackout support failed")
        require(np.all(data["confidence"][blackout] == 0) and np.all(data["forcing"][blackout] == 0), "stored patient hidden observations present")
        require(not np.any(blackout[data["contact_nodes"]]), "patient contact centre lies in withheld/guard area")
        k = (patient, mask)
        if k in base:
            for name in ("points", "triangles", "node_area", "reference_score", "bi_mv", "common_evaluation_mask"):
                require(np.array_equal(data[name], base[k][name]), f"{key}: changed original readout support or reference")
        else:
            base[k] = data
        part = frame[(frame.patient_id == patient) & (frame["mask"] == mask) & (frame.evaluation_width_mm == width) & (frame.mesh_level == level) & np.isclose(frame.pseudo_dt, dt)]
        require(set(zip(part.method, part.metric_support)) == set(itertools.product(METHODS, ("native_band", "common_3mm"))), "patient method/support grid incomplete")
        for row in part.itertuples():
            selected = support if row.metric_support == "native_band" else common
            w = area[selected]
            raw = data[row.method + "_state"]
            prediction = np.clip((raw + 1) / 2, 0, 1)[selected]
            reference = data["reference_score"][selected]
            error = prediction - reference
            mse = np.dot(w, error**2) / w.sum()
            close(row.evaluation_area_mm2, w.sum(), "patient metric changed quadrature area")
            close(row.rmse_area_weighted, np.sqrt(mse), "patient RMSE does not match stored field")
            close(row.integrated_squared_error_mm2, np.dot(w, error**2), "patient integrated error mismatch")
            truth_class, predicted_class = data["bi_mv"][selected] <= .1, prediction >= .5
            denominator = w[truth_class].sum() + w[predicted_class].sum()
            dice = 2 * w[truth_class & predicted_class].sum() / denominator if denominator else 1
            close(row.dice_voltage_le_0p1, dice, "patient Dice differs from area weighting/voltage threshold")
            if level == 1:
                full = np.load(directory / f"{key}_{row.method}.npz")["state"]
                require(np.array_equal(raw, full[:len(area)]), "fine-mesh readout is not original-node restriction")
    close(cap.capacity_absolute_error, np.abs(cap.normalised_capacity - cap.reference_normalised_capacity), "patient capacity error arithmetic")
    close(cap.capacity_absolute_log_ratio, np.abs(np.log(cap.normalised_capacity / cap.reference_normalised_capacity)), "patient capacity ratio arithmetic")
    refgroups = cap.groupby(["patient_id", "mask", "pv_label"]).reference_normalised_capacity
    require((refgroups.max() - refgroups.min()).max() < 1e-12, "patient capacity reference/domain changes with blackout width")
    summary = csv("patient_extension/patient_extension_patient_summary.csv")
    group = ["patient_id", "method", "evaluation_width_mm", "mesh_level", "pseudo_dt", "metric_support"]
    metrics = ["rmse_area_weighted", "dice_voltage_le_0p1", "mae_area_weighted"]
    expected = frame.groupby(group)[metrics].mean().sort_index()
    actual = summary.set_index(group).sort_index()
    require(actual.index.equals(expected.index), "patient aggregation omits cases")
    close(actual[metrics], expected[metrics], "patient aggregation weights vertices/PVs instead of paired masks")
    NOTES.append("Patient refinement uses the identical piecewise-planar geometry and original-node readout quadrature. It is a discretisation control, not anatomical geometry-error validation.")


def weighted_arrival(activation, target, area):
    active = target & np.isfinite(activation)
    fraction = area[active].sum() / area[target].sum()
    mean = np.dot(area[active], activation[active]) / area[active].sum() if active.any() else np.nan
    return fraction, mean


def patient_ep():
    frame = csv("patient_surface_ep_metrics.csv")
    keys = ["width_mm", "method", "dt_ms"]
    require(set(map(tuple, frame[keys].to_numpy())) == set(itertools.product((3., 10.), FOUR_METHODS, (.025, .0125))), "patient EP missing paired methods/time controls")
    unique(frame, keys, "duplicate patient EP rows")
    close(frame.t_end_ms, 210, "patient EP horizon changed")
    close(frame.d0_mm2_per_ms, .128, "patient EP baseline diffusivity differs between methods")
    require(frame.maximum_sampled_relative_linear_residual.max() < 1e-9, "patient EP algebraic residual too large")
    protocol = json.loads((DATA / "patient_surface_ep_protocol.json").read_text())
    for relative, digest in protocol["input_paths_sha256"].items():
        require(file_hash(ROOT / relative) == digest, "patient EP source hash changed")
    transfer = json.loads((DATA / "patient_surface_ep_transfer_control.json").read_text())
    require(transfer["clipped_vertices"] == 0 and transfer["activation_status_disagreements"] == 0,
            "patient EP transfer roundtrip changes phase or activation labels")
    require(transfer["maximum_activation_difference_ms"] < 1e-9, "patient EP roundtrip changes arrival times")
    geometry, acts = {}, {}
    for width in (3, 10):
        geometry[width] = dict(np.load(DATA / f"patient_surface_ep_geometry_w{width}.npz"))
        g = geometry[width]
        for name in ("points", "triangles", "vertex_area", "reference_score", "distance_LSPV_mm", "stimulus_mask", "target_mask"):
            if width == 10:
                require(np.array_equal(g[name], geometry[3][name]), "patient pacing anatomy or reference changes with blackout width")
        require(np.array_equal(g["stimulus_mask"], g["distance_LSPV_mm"] <= 2), "patient stimulus not geometry defined")
        require(np.array_equal(g["target_mask"], g["distance_LSPV_mm"] >= 20), "patient target not geometry defined")
        require(not np.any(g["stimulus_mask"] & g["target_mask"]), "patient pacing directly stimulates target")
        for method, dt in itertools.product(FOUR_METHODS, (.025, .0125)):
            key = (width, method, dt)
            with np.load(DATA / f"patient_surface_ep_w{width}_{method}_dt{dt:g}.npz") as f:
                acts[key] = f["activation"]
                require(np.isfinite(f["v"]).all() and np.isfinite(f["h"]).all(), "patient EP non-finite terminal state")
    for row in frame.itertuples():
        g = geometry[int(row.width_mm)]
        a = acts[(row.width_mm, row.method, row.dt_ms)]
        target, area = g["target_mask"].astype(bool), g["vertex_area"]
        fraction, mean = weighted_arrival(a, target, area)
        close(row.target_activation_area_fraction, fraction, "patient EP fraction not area weighted")
        close(row.target_mean_arrival_ms, mean, "patient EP mean arrival support differs")
        require(row.target_capture_90_percent == int(fraction >= .9), "patient EP capture threshold differs from 90 percent")
        ref = acts[(row.width_mm, "reference", row.dt_ms)]
        common = target & np.isfinite(a) & np.isfinite(ref)
        rmse = np.sqrt(np.dot(area[common], (a[common] - ref[common])**2) / area[common].sum())
        close(row.target_arrival_rmse_ms, rmse, "patient paired arrival uses wrong reference/support")
        common_all = target.copy()
        for method in FOUR_METHODS:
            common_all &= np.isfinite(acts[(row.width_mm, method, row.dt_ms)])
        common_rmse = np.sqrt(np.dot(area[common_all], (a[common_all] - ref[common_all])**2) / area[common_all].sum())
        close(row.target_all_method_common_area_fraction, area[common_all].sum() / area[target].sum(),
              "patient method comparison changes common activated support")
        close(row.target_arrival_common_support_rmse_ms, common_rmse, "patient common-support arrival RMSE wrong")
    mesh = csv("patient_surface_ep_mesh_control.csv")
    require(set(zip(mesh.width_mm, mesh.method)) == set(itertools.product((3, 10), ("reference", "graph"))), "patient EP mesh controls incomplete")
    for row in mesh.itertuples():
        g = geometry[int(row.width_mm)]
        n = len(g["vertex_area"])
        full = np.load(DATA / f"patient_surface_ep_w{row.width_mm:g}_{row.method}_h1_dt0.025.npz")["activation"]
        require(len(full) == row.n_vertices and row.n_triangles == 4 * len(g["triangles"]), "patient mesh refinement is not declared four-way subdivision")
        a = full[:n]
        fraction, mean = weighted_arrival(a, g["target_mask"].astype(bool), g["vertex_area"])
        close(row.target_activation_area_fraction, fraction, "mesh comparison changes quadrature/support")
        close(row.target_mean_arrival_ms, mean, "mesh mean arrival mismatch")
        r0, g0 = acts[(row.width_mm, "reference", .025)], acts[(row.width_mm, "graph", .025)]
        r1 = np.load(DATA / f"patient_surface_ep_w{row.width_mm:g}_reference_h1_dt0.025.npz")["activation"][:n]
        g1 = np.load(DATA / f"patient_surface_ep_w{row.width_mm:g}_graph_h1_dt0.025.npz")["activation"][:n]
        common = g["target_mask"].astype(bool) & np.isfinite(r0) & np.isfinite(g0) & np.isfinite(r1) & np.isfinite(g1)
        w = g["vertex_area"][common]
        close(row.native_graph_reference_rmse_common_native_refined_ms,
              np.sqrt(np.dot(w, (g0[common] - r0[common])**2) / w.sum()), "native mesh-control RMSE changes common support")
        close(row.refined_graph_reference_rmse_common_native_refined_ms,
              np.sqrt(np.dot(w, (g1[common] - r1[common])**2) / w.sum()), "refined mesh-control RMSE changes common support")
    verification = csv("patient_surface_ep_verification.csv")
    require(len(verification) == 8, "patient EP verification incomplete")
    for study, part in verification.groupby("study"):
        scale = "n" if study == "heat_spatial" else "dt_ms"
        part = part.sort_values(scale, ascending=scale == "n")
        require(np.all(np.diff(part.weighted_l2_error) < 0), f"{study}: refinement errors fail to decrease")
        require(part.observed_order.dropna().min() > (1.7 if study == "heat_spatial" else .7), f"{study}: reported convergence unsupported")
    NOTES.append(f"Patient EP stiffness is not an M-matrix; the recorded refined voltage minimum is {mesh.v_min.min():.6f}. Absolute patient EP readouts are not spatially converged.")


def certificate_and_cost():
    f = csv("graph_error_certificate.csv")
    require(len(f) == 6, "certificate table incomplete")
    for _, p in f.groupby("backend"):
        p = p.sort_values("admm_tolerance", ascending=False)
        require(np.all(np.diff(p.rms_error_bound) < 0), "algebraic certificate does not tighten")
    close(f.B_error_bound**2, 2 * f.gap, "certificate strong-convexity factor wrong", atol=1e-25)
    close(f.gap, .5 * f.stationarity_Binverse_squared + f.complementarity_defect, "certificate gap decomposition wrong", atol=1e-25)
    require(np.all(f.rms_difference_from_tight_solve <= f.rms_error_bound + f.reference_rms_error_bound + 1e-12), "certificate inconsistent with tighter-solve comparison")
    require(f.gap_identity_absolute_defect.max() < 1e-12, "primal-dual gap identity defect")
    benchmark = json.loads((DATA / "patient_extension/patient_factor_reuse_benchmark.json").read_text())
    for method in ("passive", "graph"):
        rows = [r for r in benchmark["results"] if r["method"] == method]
        require(len(rows) == 2 and {r["factor_reuse"] for r in rows} == {False, True}, "cost benchmark not paired")
        require(all(r["full_trajectory_bitwise_equal"] and r["trajectory_max_abs_difference"] == 0 for r in rows), "factor reuse changes trajectory")
        require(rows[0]["admm_iterations"] == rows[1]["admm_iterations"], "cost comparison changes graph iteration count")
        require(rows[0]["steps"] == rows[1]["steps"] == 120, "cost comparison changes horizon")
    conduction = csv("conduction_extension.csv")
    require(set(map(tuple, conduction[["dx_mm", "dt_ms", "direction"]].to_numpy())) == {
        (dx, dt, d) for (dx, dt), d in itertools.product(((.5, .01), (.25, .01), (.125, .01), (.125, .005)), ("x", "y"))}, "conduction spatial/temporal design incomplete")
    require((conduction.regression_R2 > .99).all(), "planar conduction regression is not linear")
    require((conduction.cfl < .95).all(), "conduction CFL exceeded")
    measured = conduction.dropna(subset=["cv_full_sheet_abs_difference"])
    require(len(measured) >= 4 and measured.cv_full_sheet_abs_difference.max() < 1e-10, "invariant-strip velocity differs from full sheet")


SECTIONS = {
    "synthetic": synthetic,
    "reconstruction_ep": reconstruction_ep,
    "patient_reconstruction": patient_reconstruction,
    "patient_ep": patient_ep,
    "certificate_and_cost": certificate_and_cost,
}


def main():
    global CURRENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", choices=tuple(SECTIONS), default=list(SECTIONS))
    args = parser.parse_args()
    results = []
    for name in args.sections:
        CURRENT = name
        try:
            SECTIONS[name]()
            record = {"section": name, "passed": True, "checks": COUNTS[name]}
        except Exception as exc:
            record = {"section": name, "passed": False, "checks": COUNTS.get(name, 0), "error": str(exc)}
            traceback.print_exc()
        results.append(record)
        print(json.dumps(record), flush=True)
    report = {"all_requested_sections_passed": all(r["passed"] for r in results),
              "complete_revision_validation": len(args.sections) == len(SECTIONS),
              "sections": results, "interpretation_notes": NOTES}
    (DATA / "revision_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_requested_sections_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
