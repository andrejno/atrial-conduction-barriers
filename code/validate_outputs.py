"""Check the committed numerical outputs without rerunning either solver.

The checks verify numerical identities, finite schemas, resolution metadata,
and conservation rather than enforcing a preselected scientific conclusion.
The patient experiment also checks its declared anatomical cohort and provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_csv(name: str, required: Iterable[str]) -> pd.DataFrame:
    path = DATA / name
    require(path.exists(), f"missing output: {name}")
    frame = pd.read_csv(path)
    missing = set(required).difference(frame.columns)
    require(not missing, f"{name} is missing columns: {sorted(missing)}")
    require(len(frame) > 0, f"{name} has no rows")
    numeric = frame.select_dtypes(include=[np.number])
    require(not numeric.empty, f"{name} has no numeric fields")
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        require(not bool(np.isinf(values).any()), f"{name}: {column} contains an infinite value")
        if not column.startswith("order_"):
            require(bool(np.isfinite(values).all()), f"{name}: {column} contains a non-finite value")
    return frame


def strictly_decreasing(values: pd.Series) -> bool:
    return bool(np.all(np.diff(values.to_numpy(dtype=float)) < 0.0))


def observed_orders(
    frame: pd.DataFrame,
    scale: str,
    errors: Iterable[str],
    label: str,
    coarse_to_fine_descending: bool = True,
) -> None:
    ordered = frame.sort_values(scale, ascending=not coarse_to_fine_descending)
    scale_values = ordered[scale].to_numpy(dtype=float)
    scale_step = np.diff(scale_values)
    expected_sign = -1.0 if coarse_to_fine_descending else 1.0
    require(bool(np.all(expected_sign * scale_step > 0.0)), f"{label}: {scale} levels are not distinct")
    for error in errors:
        require(strictly_decreasing(ordered[error]), f"{label}: {error} does not decrease under refinement")
        order_name = f"order_{error}"
        if order_name in ordered:
            tail = float(ordered.iloc[-1][order_name])
            require(np.isfinite(tail) and tail > 0.0, f"{label}: final {order_name} is not positive")


def validate_exact_solutions() -> None:
    phase = load_csv("phase_exact_temporal.csv", ["dt", "E0", "EL", "EH2_time", "max_mass_defect", "max_state_residual"])
    phase_spatial = load_csv("phase_exact_spatial.csv", ["n", "forcing_grid_modes", "manufactured_relative_residual"])
    phase_solution_spatial = load_csv(
        "phase_exact_solution_spatial.csv",
        [
            "n", "h", "dt", "T", "E0", "EL", "EH2_time",
            "max_mass_defect", "max_state_residual", "max_laplacian_tail",
            "order_E0", "order_EL", "order_EH2_time",
        ],
    )
    ep_temporal = load_csv("ep_exact_temporal.csv", ["n", "dx", "dt", "EV_L2", "EV_H1", "Eh_L2", "cfl"])
    ep_spatial = load_csv("ep_exact_spatial.csv", ["n", "dx", "dt", "EV_L2", "EV_H1", "Eh_L2", "cfl"])
    terms = json.loads((DATA / "phase_exact_term_balance.json").read_text())

    observed_orders(phase, "dt", ["E0", "EL", "EH2_time"], "phase temporal refinement")
    require(float(phase["max_mass_defect"].max()) < 1.0e-12, "phase mass defect is too large")
    require(float(phase["max_state_residual"].max()) < 1.0e-7, "phase solve residual is too large")
    require(bool((phase_spatial["forcing_grid_modes"] == 3 * phase_spatial["n"]).all()), "manufactured forcing lacks threefold-grid metadata")
    observed_orders(
        phase_spatial,
        "n",
        ["manufactured_relative_residual"],
        "phase manufactured residual",
        coarse_to_fine_descending=False,
    )
    require(
        len(phase_solution_spatial) == 4
        and phase_solution_spatial["n"].nunique() == 4,
        "phase solution spatial study must contain four distinct grids",
    )
    require(
        phase_solution_spatial["dt"].nunique() == 1
        and phase_solution_spatial["T"].nunique() == 1,
        "phase solution spatial study changes its time discretisation",
    )
    require(
        bool(np.allclose(
            phase_solution_spatial["h"],
            1.0 / phase_solution_spatial["n"],
            rtol=2.0e-15,
            atol=0.0,
        )),
        "phase solution spatial grid sizes are inconsistent with h=1/n",
    )
    observed_orders(
        phase_solution_spatial,
        "n",
        ["E0", "EL", "EH2_time"],
        "phase computed-solution spatial refinement",
        coarse_to_fine_descending=False,
    )
    require(
        float(phase_solution_spatial["max_mass_defect"].max()) < 1.0e-12,
        "phase spatial-study mass defect is too large",
    )
    require(
        float(phase_solution_spatial["max_state_residual"].max()) < 1.0e-7,
        "phase spatial-study solve residual is too large",
    )

    time_control_path = DATA / "phase_exact_solution_spatial_time_control.json"
    require(time_control_path.is_file(), "missing phase spatial-study time control")
    time_control = json.loads(time_control_path.read_text())
    control_metrics = ("E0", "EL", "EH2_time")
    required_control = {"n", "fine_dt", "coarse_dt"}
    for metric in control_metrics:
        required_control.update(
            {
                f"fine_{metric}", f"coarse_{metric}",
                f"solution_change_{metric}",
                f"solution_change_fraction_of_{metric}",
                f"scalar_error_change_fraction_{metric}",
            }
        )
    require(required_control.issubset(time_control), "phase time-control JSON is incomplete")
    require(
        all(np.isfinite(float(time_control[key])) for key in required_control),
        "phase time-control JSON contains a non-finite value",
    )
    finest = phase_solution_spatial.sort_values("n").iloc[-1]
    require(int(time_control["n"]) == int(finest["n"]), "phase time control uses the wrong grid")
    require(
        np.isclose(float(time_control["fine_dt"]), float(finest["dt"]), rtol=0.0, atol=1.0e-18)
        and np.isclose(
            float(time_control["coarse_dt"]),
            2.0 * float(time_control["fine_dt"]),
            rtol=0.0,
            atol=1.0e-18,
        ),
        "phase time control does not compare dt with 2dt",
    )
    for metric in control_metrics:
        require(
            np.isclose(
                float(time_control[f"fine_{metric}"]),
                float(finest[metric]),
                rtol=2.0e-13,
                atol=2.0e-15,
            ),
            f"phase time control does not reproduce finest-grid {metric}",
        )
        expected_fraction = (
            float(time_control[f"solution_change_{metric}"])
            / float(time_control[f"fine_{metric}"])
        )
        require(
            np.isclose(
                float(time_control[f"solution_change_fraction_of_{metric}"]),
                expected_fraction,
                rtol=2.0e-13,
                atol=2.0e-15,
            ),
            f"phase time-control fraction is inconsistent for {metric}",
        )
        require(
            expected_fraction < 0.02,
            f"phase spatial trend is not cleanly separated from time error in {metric}",
        )
    observed_orders(ep_temporal, "dt", ["EV_L2", "EV_H1", "Eh_L2"], "EP temporal refinement")
    observed_orders(ep_spatial, "dx", ["EV_L2", "EV_H1", "Eh_L2"], "EP spatial refinement")
    require(float(ep_temporal["cfl"].max()) < 0.95, "EP temporal study violates the explicit CFL bound")
    require(float(ep_spatial["cfl"].max()) < 0.95, "EP spatial study violates the explicit CFL bound")

    required_terms = {"time derivative", "double-well", "biharmonic", "active", "confidence", "forcing", "closure_relative_residual", "grid_modes"}
    require(required_terms.issubset(terms), "term-balance JSON is incomplete")
    require(all(np.isfinite(float(terms[key])) for key in required_terms), "term-balance JSON contains non-finite values")
    require(int(terms["grid_modes"]) > 0, "term balance has no grid metadata")
    require(float(terms["closure_relative_residual"]) < 1.0e-10, "same-grid manufactured balance does not close")
    nonzero = required_terms.difference({"closure_relative_residual", "grid_modes"})
    require(all(float(terms[key]) > 0.0 for key in nonzero), "a manufactured PDE term is unexpectedly zero")


def validate_graph_limit() -> None:
    graph = load_csv("graph_limit.csv", [
        "level", "n", "dt", "N", "EH2_time", "Echi_low", "graph_max_primal", "graph_max_dual",
        "graph_max_state", "graph_max_box", "graph_max_comp", "graph_max_projection",
    ])
    representative_path = DATA / "graph_representative.npz"
    require(representative_path.exists(), "missing graph representative")
    representative = np.load(representative_path)
    for level, group in graph.groupby("level"):
        ordered = group.sort_values("N")
        require(strictly_decreasing(ordered["EH2_time"]), f"graph state error is not monotone on {level}")
        require(strictly_decreasing(ordered["Echi_low"]), f"graph active-product error is not monotone on {level}")
    for column, tolerance in {
        "graph_max_primal": 3.0e-8, "graph_max_dual": 3.0e-8, "graph_max_state": 3.0e-8,
        "graph_max_box": 3.0e-10, "graph_max_comp": 3.0e-7, "graph_max_projection": 3.0e-7,
    }.items():
        require(float(graph[column].max()) < tolerance, f"{column} exceeds its solver tolerance")

    for key in ["laplacian", "xi", "active_mask", "projection_residual"]:
        require(key in representative.files, f"graph representative lacks {key}")
    laplacian = representative["laplacian"]
    multiplier = representative["xi"]
    active = representative["active_mask"].astype(bool)
    saved_projection = representative["projection_residual"]
    require(laplacian.shape == multiplier.shape == active.shape == saved_projection.shape, "graph representative arrays have inconsistent shapes")
    require(bool(np.isfinite(laplacian).all() and np.isfinite(multiplier).all()), "graph representative contains non-finite values")
    require(bool(np.any(active)), "graph representative has no active multiplier region")
    require(float(np.max(np.abs(multiplier[active]))) <= 1.0 + 1.0e-10, "graph multiplier violates its box constraint")
    gamma = 1.0 / max(1.0, float(np.sqrt(np.mean(laplacian**2))))
    projection = multiplier - np.clip(multiplier + gamma * laplacian, -1.0, 1.0)
    require(float(np.sqrt(np.mean(projection[active] ** 2))) < 3.0e-7, "graph multiplier fails the resolvent projection identity")
    require(float(np.max(np.abs(saved_projection[active]))) < 3.0e-7, "saved graph projection residual is too large")


def validate_calibration() -> None:
    calibration = load_csv("conduction_calibration.csv", [
        "study", "configuration", "dx_mm", "dt_ms", "direction", "cv_m_per_s", "regression_R2", "APD90_ms",
        "is_production", "cv_relative_to_finest", "apd90_relative_to_finest", "cfl",
    ])
    require(set(calibration["direction"]) == {"x", "y"}, "both fibre directions are required")
    require({"space", "time"}.issubset(set(calibration["study"])), "missing independent refinement study")
    require(float(calibration["regression_R2"].min()) > 0.999, "activation-time regression is not sufficiently linear")
    require(float(calibration["cfl"].max()) < 0.95, "calibration violates the explicit CFL bound")
    require(bool((calibration["APD90_ms"] > 0.0).all()), "APD90 interpolation returned a non-positive value")
    for direction in ["x", "y"]:
        mesh = calibration[(calibration["study"] == "space") & (calibration["direction"] == direction)]
        time = calibration[(calibration["study"] == "time") & (calibration["direction"] == direction)]
        require(mesh["dx_mm"].nunique() >= 3 and mesh["dt_ms"].nunique() == 1, f"{direction} mesh study is not an independent h refinement")
        require(time["dt_ms"].nunique() >= 3 and time["dx_mm"].nunique() == 1, f"{direction} time study is not an independent dt refinement")
    require(int(calibration["is_production"].astype(bool).sum()) == 2, "production calibration must have one run per fibre direction")


def validate_applied_outputs() -> None:
    validate_geometry_reconstruction_outputs()
    validate_pvi_outputs()


def load_applied_csv(
    name: str,
    required: Iterable[str],
    allowed_nan: Iterable[str] = (),
) -> pd.DataFrame:
    """Load an applied table and reject every undocumented non-finite field."""
    path = DATA / name
    require(path.exists(), f"missing output: {name}")
    frame = pd.read_csv(path)
    missing = set(required).difference(frame.columns)
    require(not missing, f"{name} is missing columns: {sorted(missing)}")
    require(len(frame) > 0, f"{name} has no rows")
    allowed = set(allowed_nan)
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = frame[column].to_numpy(dtype=float)
            require(not bool(np.isinf(values).any()), f"{name}: {column} contains an infinite value")
            if column not in allowed:
                require(bool(np.isfinite(values).all()), f"{name}: undocumented NaN in {column}")
        elif column not in allowed:
            require(bool(frame[column].notna().all()), f"{name}: undocumented missing value in {column}")
    return frame


def require_binary(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        require(bool(np.isin(values, [0.0, 1.0]).all()), f"{label}: {column} is not binary")


CAPACITY_COLUMNS = [
    "capacity", "normalized_capacity", "capacity_inner_flux", "capacity_outer_flux",
    "capacity_energy", "capacity_relative_residual", "capacity_flux_energy_defect",
]


def validate_capacity_identity(frame: pd.DataFrame, label: str) -> None:
    """Check the discrete Dirichlet principle and both boundary-flux identities."""
    missing = set(CAPACITY_COLUMNS).difference(frame.columns)
    require(not missing, f"{label}: missing capacity fields {sorted(missing)}")
    values = frame[CAPACITY_COLUMNS].to_numpy(dtype=float)
    require(bool(np.isfinite(values).all()), f"{label}: non-finite capacity diagnostic")
    require(bool((frame["capacity"] > 0.0).all()), f"{label}: capacity must be positive")
    require(bool((frame["normalized_capacity"] > 0.0).all()), f"{label}: normalized capacity must be positive")
    for column in ["capacity_inner_flux", "capacity_outer_flux", "capacity_energy"]:
        require(
            bool(np.allclose(frame["capacity"], frame[column], rtol=2.0e-10, atol=2.0e-13)),
            f"{label}: capacity and {column} do not agree",
        )
    require(float(frame["capacity_relative_residual"].max()) < 1.0e-10, f"{label}: capacity solve residual is too large")
    require(float(frame["capacity_flux_energy_defect"].max()) < 1.0e-9, f"{label}: capacity flux/energy defect is too large")


def validate_geometry_reconstruction_outputs() -> None:
    """Validate the current fixed-budget, geometry-level Experiment 4 only."""

    flow_nan = {
        "max_state_residual", "max_primal_residual", "max_dual_residual",
        "max_box_residual", "max_complementarity_residual",
        "max_graph_projection_residual", "max_mass_defect",
        "max_laplacian_tail_energy", "max_iterations",
    }
    blocks = load_applied_csv(
        "sparse_reconstruction_geometry_blocks.csv",
        [
            "geometry", "geometry_description", "block_index",
            "blackout_angle_deg", "acquisition_seed",
            "acquisition_coordinate_sha256", "contact_noise_draw_sha256",
            "truth_sha256", "method", "replication_unit",
            "within_geometry_repeat", "horizon_rule", "horizon", "pseudo_dt",
            "terminal_window_states", "n_training", "n_masked_grid_points",
            "masked_score_rmse", "masked_score_mae", "masked_score_bias",
            "masked_scaled_score_mse", "masked_calibration_intercept",
            "masked_calibration_slope", "masked_calibration_slope_abs_error",
            "masked_calibration_in_the_large", "masked_calibration_ece",
            "masked_score_correlation", "masked_out_of_range_fraction",
            "masked_dice", "masked_balanced_accuracy", "masked_sensitivity",
            "masked_specificity", "whole_score_rmse", "gap_count",
            "truth_gap_count", "gap_count_abs_error", "total_gap_width_mm",
            "truth_total_gap_width_mm", "total_gap_width_abs_error_mm",
            "largest_gap_width_mm", "truth_largest_gap_width_mm",
            "largest_gap_width_abs_error_mm", "normalized_capacity",
            "truth_normalized_capacity", "capacity_abs_error", *CAPACITY_COLUMNS,
            "score_min", "score_max", "reconstruction_n",
            "reconstruction_dx_mm", "fv_n", "subcells_per_axis",
            "blackout_arc_width_mm", "blackout_buffer_mm",
            "zero_confidence_fraction", "holdout_confidence_max",
            "holdout_data_forcing_max_abs", "noise_rms",
            "observation_noise_model", "observation_min", "observation_max",
            "observation_out_of_range_fraction", "screen_relative_residual",
            *flow_nan,
        ],
        allowed_nan=flow_nan,
    )
    geometry_names = {
        "complete_ring", "narrow_gap", "wide_gap", "two_gaps", "oblique_gap"
    }
    methods = {"screened", "passive", "graph"}
    block_design = {
        1: (0.0, 3101), 2: (90.0, 3103),
        3: (180.0, 3107), 4: (270.0, 3109),
    }
    require(len(blocks) == 60, "geometry stress test must contain 60 block rows")
    require(set(blocks["geometry"]) == geometry_names, "geometry set is incomplete")
    require(set(blocks["method"]) == methods, "geometry method set is incomplete")
    require(
        not blocks.duplicated(["geometry", "block_index", "method"]).any(),
        "geometry stress test contains duplicate rows",
    )
    require(
        blocks.groupby(["geometry", "block_index"])["method"]
        .apply(set).map(lambda value: value == methods).all(),
        "a geometry/block pair lacks one method",
    )
    for index, (angle, seed) in block_design.items():
        subset = blocks.loc[blocks["block_index"] == index]
        require(
            bool((subset["blackout_angle_deg"] == angle).all())
            and bool((subset["acquisition_seed"] == seed).all()),
            f"geometry block {index} has the wrong angle/seed pair",
        )
        require(
            subset["acquisition_coordinate_sha256"].nunique() == 1
            and subset["contact_noise_draw_sha256"].nunique() == 1,
            f"geometry block {index} does not reuse its acquisition across geometries",
        )
    for column in (
        "acquisition_coordinate_sha256", "contact_noise_draw_sha256", "truth_sha256"
    ):
        require(
            bool(blocks[column].str.fullmatch(r"[0-9a-f]{64}").all()),
            f"geometry {column} is not a SHA-256 digest",
        )

    require(bool((blocks["replication_unit"] == "geometry").all()), "wrong replication unit")
    require(
        bool((blocks["within_geometry_repeat"] == "fixed_spatial_block").all()),
        "within-geometry repeats are not fixed spatial blocks",
    )
    require(
        bool((blocks["horizon_rule"] == "fixed_common_120_step_budget_no_selection").all()),
        "geometry experiment does not use the fixed computational budget",
    )
    dynamic = blocks["method"].isin(["passive", "graph"])
    require(bool(np.allclose(blocks.loc[dynamic, "horizon"], 1.20)), "dynamic horizons differ")
    require(bool((blocks.loc[~dynamic, "horizon"] == 0.0).all()), "screened horizon is nonzero")
    require(bool(np.allclose(blocks["pseudo_dt"], 0.01)), "wrong geometry pseudo-time step")
    require(
        bool((blocks.loc[dynamic, "terminal_window_states"] == 12).all())
        and bool((blocks.loc[~dynamic, "terminal_window_states"] == 0).all()),
        "geometry terminal averaging is inconsistent",
    )
    require(bool((blocks["n_masked_grid_points"] > 0).all()), "an evaluation block is empty")
    require(bool((blocks["holdout_confidence_max"] == 0.0).all()), "held-out confidence leaks")
    require(bool((blocks["holdout_data_forcing_max_abs"] == 0.0).all()), "held-out forcing leaks")
    require(float(blocks["screen_relative_residual"].max()) < 2.0e-9, "screened solve is inaccurate")

    # The uncensored law is observable in the saved data: at least one score in
    # every acquisition lies outside the bounded exact-score range, while the
    # realised RMS remains close to the prescribed 0.12 standard deviation.
    require(
        bool((blocks["observation_noise_model"] == "uncensored_gaussian").all()),
        "geometry observation law is not declared uncensored Gaussian",
    )
    require(
        bool(((blocks["observation_out_of_range_fraction"] > 0.0)
              & (blocks["observation_out_of_range_fraction"] < 1.0)).all()),
        "saved geometry observations do not demonstrate the uncensored law",
    )
    require(
        bool(((blocks["noise_rms"] > 0.09) & (blocks["noise_rms"] < 0.15)).all()),
        "realised geometry noise is inconsistent with SD 0.12",
    )
    observation_invariants = [
        "n_training", "n_masked_grid_points", "zero_confidence_fraction",
        "screen_relative_residual", "acquisition_coordinate_sha256",
        "contact_noise_draw_sha256", "observation_noise_model",
        "observation_min", "observation_max", "observation_out_of_range_fraction",
    ]
    require(
        bool((blocks.groupby(["geometry", "block_index"])[observation_invariants]
              .nunique(dropna=False) == 1).all().all()),
        "methods do not share identical geometry/block observations",
    )

    require(
        bool(np.allclose(
            blocks["masked_scaled_score_mse"],
            0.25 * blocks["masked_score_rmse"] ** 2,
            rtol=2.0e-12,
            atol=2.0e-14,
        )),
        "scaled-score MSE is inconsistent with score RMSE",
    )
    for column in (
        "masked_calibration_ece", "masked_out_of_range_fraction", "masked_dice",
        "masked_balanced_accuracy", "masked_sensitivity", "masked_specificity",
    ):
        require(
            bool(((blocks[column] >= 0.0) & (blocks[column] <= 1.0 + 1.0e-12)).all()),
            f"geometry {column} lies outside its admissible range",
        )
    for measured, truth, error in (
        ("gap_count", "truth_gap_count", "gap_count_abs_error"),
        ("total_gap_width_mm", "truth_total_gap_width_mm", "total_gap_width_abs_error_mm"),
        ("largest_gap_width_mm", "truth_largest_gap_width_mm", "largest_gap_width_abs_error_mm"),
        ("normalized_capacity", "truth_normalized_capacity", "capacity_abs_error"),
    ):
        require(
            bool(np.allclose(blocks[error], np.abs(blocks[measured] - blocks[truth]))),
            f"geometry {error} is inconsistent",
        )
    expected_truth_counts = {
        "complete_ring": 0, "narrow_gap": 1, "wide_gap": 1,
        "two_gaps": 2, "oblique_gap": 1,
    }
    for geometry, expected in expected_truth_counts.items():
        require(
            bool((blocks.loc[blocks["geometry"] == geometry, "truth_gap_count"] == expected).all()),
            f"{geometry}: hidden gap count is wrong",
        )
    truth_columns = [
        "truth_gap_count", "truth_total_gap_width_mm", "truth_largest_gap_width_mm",
        "truth_normalized_capacity", "truth_sha256",
    ]
    require(
        bool((blocks.groupby("geometry")[truth_columns].nunique(dropna=False) == 1).all().all()),
        "a hidden geometry changes across blocks or methods",
    )
    validate_capacity_identity(blocks, "geometry-replicated reconstruction")
    graph = blocks.loc[blocks["method"] == "graph"]
    for column, tolerance in {
        "max_state_residual": 1.0e-6,
        "max_primal_residual": 1.0e-6,
        "max_dual_residual": 1.0e-6,
        "max_box_residual": 1.0e-10,
        "max_complementarity_residual": 2.0e-6,
        "max_graph_projection_residual": 1.0e-6,
        "max_mass_defect": 1.0e-12,
    }.items():
        require(float(graph[column].max()) < tolerance, f"geometry graph {column} exceeds tolerance")

    units = load_applied_csv(
        "sparse_reconstruction_geometry_units.csv",
        [
            "geometry", "geometry_description", "method", "replication_unit",
            "n_spatial_blocks_averaged", "block_averaging_rule", "horizon_rule",
            "horizon", "masked_score_rmse", "masked_score_mae", "masked_score_bias",
            "masked_scaled_score_mse", "masked_dice", "masked_balanced_accuracy",
            "whole_score_rmse", "capacity_abs_error", "gap_count_abs_error",
            "total_gap_width_abs_error_mm", "largest_gap_width_abs_error_mm",
            "masked_score_rmse_within_block_sd", "masked_calibration_intercept",
            "masked_calibration_slope", "masked_calibration_slope_abs_error",
            "masked_calibration_in_the_large", "masked_calibration_ece",
            "masked_score_correlation", "masked_out_of_range_fraction",
        ],
    )
    require(
        len(units) == 15 and not units.duplicated(["geometry", "method"]).any(),
        "geometry-level table must contain one row per geometry and method",
    )
    require(set(units["geometry"]) == geometry_names and set(units["method"]) == methods, "incomplete geometry units")
    require(bool((units["replication_unit"] == "geometry").all()), "unit table changes replication unit")
    require(bool((units["n_spatial_blocks_averaged"] == 4).all()), "unit table does not average four blocks")
    require(
        bool((units["block_averaging_rule"] == "arithmetic_mean_before_method_contrast").all()),
        "unit table uses the wrong block averaging rule",
    )
    averaged = [
        "masked_score_rmse", "masked_score_mae", "masked_score_bias",
        "masked_scaled_score_mse", "masked_dice", "masked_balanced_accuracy",
        "whole_score_rmse", "capacity_abs_error", "gap_count_abs_error",
        "total_gap_width_abs_error_mm", "largest_gap_width_abs_error_mm",
    ]
    direct_units = blocks.groupby(["geometry", "method"])[averaged].mean()
    for row in units.itertuples(index=False):
        expected = direct_units.loc[(row.geometry, row.method)]
        for column in averaged:
            require(
                bool(np.isclose(getattr(row, column), expected[column], rtol=2.0e-12, atol=2.0e-14)),
                f"{row.geometry}/{row.method}: {column} is not the four-block mean",
            )

    summary = load_applied_csv(
        "sparse_reconstruction_geometry_summary.csv",
        [
            "metric", "n_geometries", "screened_mean", "passive_mean", "graph_mean",
            "graph_minus_passive_mean", "graph_minus_passive_median",
            "graph_minus_passive_min", "graph_minus_passive_max",
            "graph_better_geometry_count", "tied_geometry_count",
            "graph_worse_geometry_count", "interval_type",
        ],
    )
    summary_metrics = {
        "masked_score_rmse", "masked_scaled_score_mse", "masked_calibration_ece",
        "masked_calibration_slope_abs_error", "capacity_abs_error",
        "gap_count_abs_error", "largest_gap_width_abs_error_mm",
    }
    require(len(summary) == 7 and set(summary["metric"]) == summary_metrics, "wrong geometry summary metrics")
    require(bool((summary["n_geometries"] == 5).all()), "geometry summary uses pseudoreplicates")
    require(
        bool((summary["interval_type"] == "none_fixed_prespecified_scenario_set").all()),
        "geometry summary reports an unsupported population interval",
    )
    for row in summary.itertuples(index=False):
        pivot = units.pivot(index="geometry", columns="method", values=row.metric)
        difference = (pivot["graph"] - pivot["passive"]).to_numpy(dtype=float)
        expected = {
            "screened_mean": pivot["screened"].mean(),
            "passive_mean": pivot["passive"].mean(),
            "graph_mean": pivot["graph"].mean(),
            "graph_minus_passive_mean": np.mean(difference),
            "graph_minus_passive_median": np.median(difference),
            "graph_minus_passive_min": np.min(difference),
            "graph_minus_passive_max": np.max(difference),
            "graph_better_geometry_count": np.sum(difference < -1.0e-14),
            "tied_geometry_count": np.sum(np.abs(difference) <= 1.0e-14),
            "graph_worse_geometry_count": np.sum(difference > 1.0e-14),
        }
        for column, value in expected.items():
            require(
                bool(np.isclose(float(getattr(row, column)), float(value), rtol=2.0e-12, atol=2.0e-14)),
                f"geometry summary has the wrong {column} for {row.metric}",
            )

    calibration = load_applied_csv(
        "sparse_reconstruction_geometry_calibration.csv",
        [
            "scope", "geometry", "method", "bin", "bin_lower", "bin_upper",
            "n", "mean_predicted", "mean_truth",
        ],
        allowed_nan={"mean_predicted", "mean_truth"},
    )
    require(len(calibration) == 180, "reliability table must contain 180 fixed bins")
    require(set(calibration["scope"]) == {"geometry", "pooled_descriptive"}, "wrong calibration scopes")
    require(set(calibration["method"]) == methods, "incomplete calibration methods")
    require(bool(((calibration["bin"] >= 1) & (calibration["bin"] <= 10)).all()), "invalid reliability bin")
    nonempty = calibration["n"] > 0
    require(bool(calibration.loc[nonempty, ["mean_predicted", "mean_truth"]].notna().all().all()), "nonempty reliability bin omits means")
    require(bool(calibration.loc[~nonempty, ["mean_predicted", "mean_truth"]].isna().all().all()), "empty reliability bin contains means")
    expected_points = blocks.groupby(["geometry", "method"])["n_masked_grid_points"].sum()
    observed_points = calibration.loc[calibration["scope"] == "geometry"].groupby(["geometry", "method"])["n"].sum()
    require(bool((observed_points == expected_points).all()), "reliability bins omit masked scores")

    maps_path = DATA / "sparse_reconstruction_geometry_maps.npz"
    require(maps_path.is_file(), "missing geometry reconstruction maps")
    maps = np.load(maps_path)
    required_maps = {
        *(f"truth_{name}" for name in geometry_names),
        "representative_truth", "representative_confidence",
        "representative_holdout_mask", "representative_train_x",
        "representative_train_y", "representative_screened",
        "representative_passive", "representative_graph",
    }
    require(required_maps.issubset(maps.files), "geometry map archive is incomplete")
    held_out = maps["representative_holdout_mask"].astype(bool)
    require(bool(np.all(maps["representative_confidence"][held_out] == 0.0)), "representative map leaks confidence")


# RETIRED LEGACY VALIDATOR -------------------------------------------------
# Kept only to make old result files inspectable by developers.  The current
# publication validator never calls it because those files used a censored
# observation law and belong to the superseded synthetic end-to-end design.
def _validate_retired_censored_reconstruction_outputs() -> None:
    horizon = load_applied_csv(
        "reconstruction_horizon_selection.csv",
        [
            "method", "seed", "horizon", "validation_rmse", "n_training", "n_validation",
            "reconstruction_n", "reconstruction_dx_mm", "kx_weight", "ky_weight",
            "kernel_support_mm", "kernel_weight", "blackout_arc_width_mm", "blackout_angle_deg",
            "blackout_buffer_mm", "contact_noise_sd", "zero_confidence_fraction",
            "holdout_confidence_max", "holdout_data_forcing_max_abs", "screen_relative_residual",
            "max_state_residual", "mean", "standard_error", "one_se_threshold",
            "selected_horizon", "selection_rule",
        ],
    )
    require(set(horizon["method"]) == {"passive", "graph"}, "horizon selection must cover passive and graph flows")
    require(bool((horizon["horizon"] > 0.0).all()), "candidate horizons must be positive")
    require(bool((horizon["validation_rmse"] >= 0.0).all()), "validation RMSE must be non-negative")
    require(bool((horizon["n_training"] > 0).all() and (horizon["n_validation"] > 0).all()), "training/validation counts must be positive")
    require(bool(((horizon["zero_confidence_fraction"] > 0.0) & (horizon["zero_confidence_fraction"] < 1.0)).all()), "blackout occupies an invalid fraction of the grid")
    require(bool((horizon["holdout_confidence_max"] == 0.0).all()), "confidence is not exactly zero in the buffered blackout")
    require(bool((horizon["holdout_data_forcing_max_abs"] == 0.0).all()), "data forcing is not exactly zero in the buffered blackout")
    require(float(horizon["screen_relative_residual"].max()) < 2.0e-9, "screened initialization is not solved accurately")
    require(horizon.groupby("method")["seed"].apply(frozenset).nunique() == 1, "passive and graph horizon selection use different calibration seeds")

    selected_horizons: dict[str, float] = {}
    for method, group in horizon.groupby("method"):
        candidates_per_seed = group.groupby("seed")["horizon"].apply(lambda values: tuple(sorted(values)))
        require(candidates_per_seed.nunique() == 1, f"{method}: calibration seeds use different horizon candidates")
        require(len(candidates_per_seed.iloc[0]) >= 3, f"{method}: too few horizon candidates")
        summary = group[["horizon", "mean", "standard_error", "one_se_threshold", "selected_horizon"]].drop_duplicates().sort_values("horizon")
        require(len(summary) == len(candidates_per_seed.iloc[0]), f"{method}: inconsistent saved horizon summary")
        minimum = summary.loc[summary["mean"].idxmin()]
        expected_threshold = float(minimum["mean"] + minimum["standard_error"])
        require(bool(np.allclose(summary["one_se_threshold"], expected_threshold, rtol=1.0e-12, atol=1.0e-14)), f"{method}: incorrect one-standard-error threshold")
        eligible = summary.loc[summary["mean"] <= expected_threshold + 1.0e-14, "horizon"]
        expected_selected = float(eligible.min())
        require(summary["selected_horizon"].nunique() == 1, f"{method}: multiple selected horizons")
        selected = float(summary["selected_horizon"].iloc[0])
        require(np.isclose(selected, expected_selected), f"{method}: saved horizon does not implement the one-standard-error rule")
        require(bool(group["selection_rule"].str.contains("one SE", regex=False).all()), f"{method}: horizon-selection rule is not documented")
        selected_horizons[method] = selected

    maps_path = DATA / "sparse_reconstruction_maps.npz"
    require(maps_path.exists(), "missing sparse reconstruction maps")
    maps = np.load(maps_path)
    required_maps = {
        "truth", "truth_diffusivity", "truth_capacity_potential", "screened", "passive", "graph",
        "confidence", "data_forcing", "holdout_mask", "train_x", "train_y", "train_score",
        "validation_x", "validation_y",
    }
    require(required_maps.issubset(maps.files), f"sparse reconstruction maps lack {sorted(required_maps.difference(maps.files))}")
    reconstruction_shape = maps["truth"].shape
    require(len(reconstruction_shape) == 2 and reconstruction_shape[0] == reconstruction_shape[1], "reconstruction map is not square")
    for key in ["truth", "screened", "passive", "graph", "confidence", "data_forcing", "holdout_mask"]:
        require(maps[key].shape == reconstruction_shape, f"sparse reconstruction map {key} has the wrong shape")
        require(bool(np.isfinite(maps[key]).all()), f"sparse reconstruction map {key} contains a non-finite value")
    holdout_mask = maps["holdout_mask"]
    require(bool(np.isin(holdout_mask, [0.0, 1.0]).all()) and bool(np.any(holdout_mask == 1.0)), "holdout mask is invalid")
    held_out = holdout_mask.astype(bool)
    require(bool(np.all(maps["confidence"][held_out] == 0.0)), "map confidence is not exactly zero in the blackout")
    require(bool(np.all(maps["data_forcing"][held_out] == 0.0)), "map forcing is not exactly zero in the blackout")
    require(bool(np.any(maps["confidence"][~held_out] > 0.0)), "confidence is zero outside the blackout as well")

    ensemble_nan = {
        "max_state_residual", "max_primal_residual", "max_dual_residual", "max_box_residual",
        "max_complementarity_residual", "max_graph_projection_residual", "max_mass_defect",
        "max_laplacian_tail_energy", "max_iterations",
    }
    ensemble = load_applied_csv(
        "sparse_reconstruction_ensemble.csv",
        [
            "seed", "method", "selected_horizon", "rmse", "dice", "jaccard", "gap_width_mm",
            "truth_zero_level_gap_width_mm", "gap_width_abs_error_mm", "gap_region_rmse",
            "normalized_capacity", "truth_normalized_capacity", "capacity_abs_error", *CAPACITY_COLUMNS,
            "fv_n", "fv_dx_mm", "subcells_per_axis", "validation_rmse", "n_training",
            "n_validation", "noise_rms", "contact_noise_sd", "reconstruction_n",
            "reconstruction_dx_mm", "zero_confidence_fraction", "holdout_confidence_max",
            "holdout_data_forcing_max_abs", "screen_relative_residual", "screen_min", "screen_max",
            *ensemble_nan,
        ],
        ensemble_nan,
    )
    require(set(ensemble["method"]) == {"screened", "passive", "graph"}, "reconstruction ensemble has an incomplete method set")
    require(ensemble.groupby("seed")["method"].apply(set).map(lambda values: values == {"screened", "passive", "graph"}).all(), "not every reconstruction seed has all three methods")
    require(bool((ensemble["holdout_confidence_max"] == 0.0).all()), "ensemble confidence is not exactly zero in the blackout")
    require(bool((ensemble["holdout_data_forcing_max_abs"] == 0.0).all()), "ensemble forcing is not exactly zero in the blackout")
    require(bool(((ensemble["dice"] >= 0.0) & (ensemble["dice"] <= 1.0)).all()), "Dice score lies outside [0,1]")
    require(bool(((ensemble["jaccard"] >= 0.0) & (ensemble["jaccard"] <= 1.0)).all()), "Jaccard score lies outside [0,1]")
    require(bool(np.allclose(ensemble["gap_width_abs_error_mm"], np.abs(ensemble["gap_width_mm"] - ensemble["truth_zero_level_gap_width_mm"]))), "saved gap-width errors are inconsistent")
    require(bool(np.allclose(ensemble["capacity_abs_error"], np.abs(ensemble["normalized_capacity"] - ensemble["truth_normalized_capacity"]))), "saved capacity errors are inconsistent")
    validate_capacity_identity(ensemble, "sparse reconstruction ensemble")
    require(bool((ensemble.loc[ensemble["method"] == "screened", "selected_horizon"] == 0.0).all()), "screened baseline has a nonzero pseudo-time horizon")
    for method in ["passive", "graph"]:
        require(bool(np.allclose(ensemble.loc[ensemble["method"] == method, "selected_horizon"], selected_horizons[method])), f"{method}: ensemble does not use its selected horizon")
    screened = ensemble["method"] == "screened"
    require(bool(ensemble.loc[screened, list(ensemble_nan)].isna().all().all()), "screened rows contain method-inapplicable flow diagnostics")
    require(bool(ensemble.loc[~screened, list(ensemble_nan)].notna().all().all()), "flow rows omit solver diagnostics")
    graph = ensemble.loc[ensemble["method"] == "graph"]
    for column, tolerance in {
        "max_state_residual": 1.0e-6, "max_primal_residual": 1.0e-6,
        "max_dual_residual": 1.0e-6, "max_box_residual": 1.0e-10,
        "max_complementarity_residual": 2.0e-6, "max_graph_projection_residual": 1.0e-6,
        "max_mass_defect": 1.0e-12,
    }.items():
        require(float(graph[column].max()) < tolerance, f"reconstruction graph flow: {column} exceeds tolerance")

    refinement = load_applied_csv(
        "sparse_reconstruction_refinement.csv",
        [
            "study", "n", "dx_mm", "pseudo_dt", "selected_horizon", "terminal_window",
            "field_rmse", "h2k_error", "l2_difference_from_finest", "h2k_difference_from_finest",
            "gap_width_mm", "normalized_capacity", "final_laplacian_tail_energy",
            "max_laplacian_tail_energy", "max_graph_state_residual",
        ],
    )
    require(set(refinement["study"]) == {"spatial", "pseudo_time_step"}, "reconstruction lacks independent space/time-step checks")
    spatial = refinement.loc[refinement["study"] == "spatial"].sort_values("n")
    pseudo_time = refinement.loc[refinement["study"] == "pseudo_time_step"].sort_values("pseudo_dt", ascending=False)
    require(spatial["n"].nunique() >= 3 and spatial["pseudo_dt"].nunique() == 1, "reconstruction spatial study changes both grid and pseudo-time step")
    require(pseudo_time["pseudo_dt"].nunique() >= 3 and pseudo_time["n"].nunique() == 1, "reconstruction pseudo-time study changes both grid and step")
    require(bool(np.all(np.diff(spatial["dx_mm"]) < 0.0)), "reconstruction dx does not decrease with n")
    require(strictly_decreasing(spatial["l2_difference_from_finest"]), "reconstruction L2 grid difference is not monotone")
    require(strictly_decreasing(spatial["h2k_difference_from_finest"]), "reconstruction H2_K grid difference is not monotone")
    require(strictly_decreasing(pseudo_time["l2_difference_from_finest"]), "reconstruction L2 step difference is not monotone")
    require(bool(np.allclose(refinement["selected_horizon"], selected_horizons["graph"])), "graph refinement uses the wrong selected horizon")
    require(float(refinement["max_graph_state_residual"].max()) < 1.0e-6, "graph reconstruction refinement exceeds solver tolerance")

    sensitivity = load_applied_csv(
        "sparse_reconstruction_sensitivity.csv",
        [
            "initialisation", "horizon_multiplier", "horizon", "rmse", "dice",
            "validation_rmse", "normalized_capacity", "capacity_abs_error",
            "last_step_rms_change", "max_state_residual",
        ],
    )
    require(sensitivity["initialisation"].nunique() >= 2, "initialization sensitivity has fewer than two initial states")
    require(set(np.round(sensitivity["horizon_multiplier"], 12)).issuperset({0.5, 1.0, 2.0}), "horizon sensitivity does not bracket the selected value")
    require(bool(np.allclose(sensitivity["horizon"], sensitivity["horizon_multiplier"] * selected_horizons["graph"])), "sensitivity horizons are inconsistent with the selected graph horizon")
    require(float(sensitivity["max_state_residual"].max()) < 1.0e-6, "sensitivity graph solve exceeds tolerance")

    common = load_applied_csv(
        "sparse_reconstruction_common_horizon.csv",
        [
            "seed", "method", "common_horizon", "is_method_selected_horizon",
            "comparison_design", "pseudo_dt", "terminal_window_states",
            "terminal_window_pseudo_time", "rmse", "dice", "jaccard",
            "boundary_mean_mm", "boundary_hd95_mm", "ring_viable_component_count",
            "complete_barrier_indicator", "truth_ring_viable_component_count",
            "ring_topology_correct", "gap_width_mm", "truth_zero_level_gap_width_mm",
            "gap_width_abs_error_mm", "gap_region_rmse", "normalized_capacity",
            "truth_normalized_capacity", "capacity_abs_error", *CAPACITY_COLUMNS,
            "fv_n", "fv_dx_mm", "subcells_per_axis", "validation_rmse",
            "n_training", "n_validation", "noise_rms", "contact_noise_sd",
            "reconstruction_n", "reconstruction_dx_mm", "kx_weight", "ky_weight",
            "kernel_support_mm", "kernel_weight", "blackout_arc_width_mm",
            "blackout_angle_deg", "blackout_buffer_mm", "zero_confidence_fraction",
            "holdout_confidence_max", "holdout_data_forcing_max_abs",
            "screen_iterations", "screen_relative_residual", "screen_min", "screen_max",
            *ensemble_nan,
            "paired_graph_minus_passive_rmse", "paired_graph_minus_passive_dice",
            "paired_graph_minus_passive_boundary_hd95_mm",
            "paired_graph_minus_passive_gap_width_abs_error_mm",
            "paired_graph_minus_passive_gap_region_rmse",
            "paired_graph_minus_passive_normalized_capacity",
            "paired_graph_minus_passive_capacity_abs_error",
            "paired_graph_minus_passive_validation_rmse",
        ],
    )

    # Structural and arithmetic checks for the fixed, paired sensitivity design.
    locked_test_seeds = {3, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71}
    common_horizons = {0.60, 1.20}
    require(len(common) == 64, "common-horizon comparison must contain 64 rows")
    require(set(common["seed"].astype(int)) == locked_test_seeds, "common-horizon comparison does not use all 16 locked test seeds")
    require(set(common["method"]) == {"passive", "graph"}, "common-horizon comparison has an incomplete method set")
    require(set(np.round(common["common_horizon"], 12)) == common_horizons, "common-horizon comparison does not use T=0.60 and T=1.20")
    require(not common.duplicated(["seed", "common_horizon", "method"]).any(), "common-horizon comparison contains duplicate paired rows")
    require(
        common.groupby(["seed", "common_horizon"])["method"].apply(set).map(
            lambda values: values == {"passive", "graph"}
        ).all(),
        "a common-horizon seed/time pair is missing one method",
    )
    require(bool((common["comparison_design"] == "same_seed_same_horizon").all()), "common-horizon pairing is not documented")
    require_binary(
        common,
        ["is_method_selected_horizon", "complete_barrier_indicator", "ring_topology_correct"],
        "common-horizon comparison",
    )
    expected_selected = np.asarray(
        [
            np.isclose(horizon_value, selected_horizons[method])
            for method, horizon_value in zip(common["method"], common["common_horizon"])
        ],
        dtype=int,
    )
    require(
        bool((common["is_method_selected_horizon"].to_numpy(dtype=int) == expected_selected).all()),
        "common-horizon selected-horizon flags are inconsistent",
    )
    require(bool(np.allclose(common["pseudo_dt"], 0.01)), "common-horizon comparison uses the wrong pseudo-time step")
    require(bool((common["terminal_window_states"] == 12).all()), "common-horizon comparison uses the wrong terminal window")
    require(bool(np.allclose(common["terminal_window_pseudo_time"], 0.12)), "common-horizon terminal-window duration is inconsistent")
    require(bool((common["reconstruction_n"] == 81).all()), "common-horizon reconstruction grid is not the production grid")
    require(bool((common["fv_n"] == 121).all()), "common-horizon capacity grid is not the production grid")
    require(bool((common["subcells_per_axis"] == 3).all()), "common-horizon comparison lacks production subcell averaging")
    require(bool((common["holdout_confidence_max"] == 0.0).all()), "common-horizon confidence is not exactly zero in the blackout")
    require(bool((common["holdout_data_forcing_max_abs"] == 0.0).all()), "common-horizon forcing is not exactly zero in the blackout")
    require(float(common["screen_relative_residual"].max()) < 2.0e-9, "common-horizon screened initialization is not solved accurately")
    require(bool(((common["dice"] >= 0.0) & (common["dice"] <= 1.0)).all()), "common-horizon Dice lies outside [0,1]")
    require(bool(((common["jaccard"] >= 0.0) & (common["jaccard"] <= 1.0)).all()), "common-horizon Jaccard lies outside [0,1]")
    require(bool((common["ring_viable_component_count"] >= 0.0).all()), "common-horizon topology has a negative component count")
    require(bool(np.allclose(common["ring_viable_component_count"], np.round(common["ring_viable_component_count"]))), "common-horizon topology component count is not integral")
    require(bool(np.allclose(common["gap_width_abs_error_mm"], np.abs(common["gap_width_mm"] - common["truth_zero_level_gap_width_mm"]))), "common-horizon gap-width errors are inconsistent")
    require(bool(np.allclose(common["capacity_abs_error"], np.abs(common["normalized_capacity"] - common["truth_normalized_capacity"]))), "common-horizon capacity errors are inconsistent")
    validate_capacity_identity(common, "common-horizon comparison")
    common_graph = common.loc[common["method"] == "graph"]
    for column, tolerance in {
        "max_state_residual": 1.0e-6, "max_primal_residual": 1.0e-6,
        "max_dual_residual": 1.0e-6, "max_box_residual": 1.0e-10,
        "max_complementarity_residual": 2.0e-6,
        "max_graph_projection_residual": 1.0e-6, "max_mass_defect": 1.0e-12,
    }.items():
        require(float(common_graph[column].max()) < tolerance, f"common-horizon graph flow: {column} exceeds tolerance")

    seed_invariants = [
        "n_training", "n_validation", "noise_rms", "contact_noise_sd",
        "zero_confidence_fraction", "screen_iterations", "screen_relative_residual",
        "screen_min", "screen_max",
    ]
    require(
        bool((common.groupby("seed")[seed_invariants].nunique(dropna=False) == 1).all().all()),
        "common-horizon methods do not share the same seed-specific observations and initialization",
    )
    truth_invariants = [
        "truth_zero_level_gap_width_mm", "truth_normalized_capacity",
        "truth_ring_viable_component_count",
    ]
    require(
        bool((common[truth_invariants].nunique(dropna=False) == 1).all()),
        "common-horizon hidden truth changes across rows",
    )

    paired_metrics = [
        "rmse", "dice", "boundary_hd95_mm", "gap_width_abs_error_mm",
        "gap_region_rmse", "normalized_capacity", "capacity_abs_error",
        "validation_rmse",
    ]
    for metric in paired_metrics:
        pivot = common.pivot(index=["seed", "common_horizon"], columns="method", values=metric)
        expected_difference = (pivot["graph"] - pivot["passive"]).to_dict()
        repeated = np.asarray(
            [expected_difference[(seed, time)] for seed, time in zip(common["seed"], common["common_horizon"])],
            dtype=float,
        )
        saved = common[f"paired_graph_minus_passive_{metric}"].to_numpy(dtype=float)
        require(bool(np.allclose(saved, repeated, rtol=2.0e-13, atol=2.0e-15)), f"common-horizon paired {metric} contrast is inconsistent")

    # Regression check: the rows at each method's locked calibration horizon
    # must reproduce the previously committed Experiment 4 ensemble exactly.
    selected_common = common.loc[common["is_method_selected_horizon"] == 1]
    selected_ensemble = ensemble.loc[ensemble["method"].isin(["passive", "graph"])]
    compared = selected_common.merge(
        selected_ensemble,
        on=["seed", "method"],
        how="inner",
        validate="one_to_one",
        suffixes=("_common", "_ensemble"),
    )
    require(len(compared) == 32, "regression check: selected common-horizon rows do not match all ensemble keys")
    shared_numeric = [
        column for column in common.columns
        if column in ensemble.columns and column not in {"seed", "method"}
        and pd.api.types.is_numeric_dtype(common[column])
    ]
    for column in shared_numeric:
        require(
            bool(np.allclose(
                compared[f"{column}_common"], compared[f"{column}_ensemble"],
                rtol=2.0e-12, atol=2.0e-14, equal_nan=True,
            )),
            f"regression check: selected-horizon {column} does not reproduce the ensemble",
        )

    # Claim-consistency checks for the two prespecified primary endpoints.
    for metric in ["rmse", "capacity_abs_error"]:
        paired_column = f"paired_graph_minus_passive_{metric}"
        require(
            bool((common[paired_column] < 0.0).all()),
            f"claim consistency: graph does not lower paired {metric} in every seed at both common horizons",
        )

    common_summary = load_applied_csv(
        "sparse_reconstruction_common_horizon_summary.csv",
        [
            "common_horizon", "metric", "n", "passive_mean", "graph_mean",
            "graph_minus_passive_mean", "graph_minus_passive_sample_sd",
            "interval_type", "lower_95", "upper_95", "bootstrap_resamples",
            "bootstrap_seed", "graph_lower_count", "tied_count",
            "graph_higher_count",
        ],
    )
    summary_metrics = {
        "rmse", "capacity_abs_error", "gap_width_abs_error_mm", "gap_region_rmse"
    }
    require(len(common_summary) == 8, "common-horizon summary must contain eight rows")
    require(
        set(np.round(common_summary["common_horizon"], 12)) == common_horizons,
        "common-horizon summary has the wrong horizons",
    )
    require(
        set(common_summary["metric"]) == summary_metrics,
        "common-horizon summary has the wrong metrics",
    )
    require(
        not common_summary.duplicated(["common_horizon", "metric"]).any(),
        "common-horizon summary has duplicate horizon/metric rows",
    )
    require(
        bool((common_summary["interval_type"] == "paired_bootstrap_percentile_mean").all()),
        "common-horizon summary uses the wrong interval type",
    )
    require(
        bool((common_summary["bootstrap_resamples"] == 5000).all())
        and bool((common_summary["bootstrap_seed"] == 20260903).all()),
        "common-horizon summary has the wrong bootstrap design",
    )
    for row in common_summary.itertuples(index=False):
        subset = common.loc[np.isclose(common["common_horizon"], row.common_horizon)]
        pivot = subset.pivot(index="seed", columns="method", values=row.metric)
        differences = (pivot["graph"] - pivot["passive"]).to_numpy(dtype=float)
        rng = np.random.default_rng(20260903)
        means = np.mean(
            rng.choice(differences, size=(5000, len(differences)), replace=True), axis=1
        )
        expected_lower, expected_upper = np.quantile(means, (0.025, 0.975))
        checks = {
            "n": (int(row.n), len(differences)),
            "passive mean": (float(row.passive_mean), float(pivot["passive"].mean())),
            "graph mean": (float(row.graph_mean), float(pivot["graph"].mean())),
            "paired mean": (float(row.graph_minus_passive_mean), float(np.mean(differences))),
            "paired sample SD": (
                float(row.graph_minus_passive_sample_sd),
                float(np.std(differences, ddof=1)),
            ),
            "lower interval": (float(row.lower_95), float(expected_lower)),
            "upper interval": (float(row.upper_95), float(expected_upper)),
            "graph-lower count": (int(row.graph_lower_count), int(np.sum(differences < 0.0))),
            "tie count": (int(row.tied_count), int(np.sum(differences == 0.0))),
            "graph-higher count": (int(row.graph_higher_count), int(np.sum(differences > 0.0))),
        }
        for label, (saved, expected) in checks.items():
            require(
                bool(np.isclose(saved, expected, rtol=2.0e-12, atol=2.0e-14)),
                f"common-horizon summary has the wrong {label} for T={row.common_horizon:g}, {row.metric}",
            )

    geometry_blocks = load_applied_csv(
        "sparse_reconstruction_geometry_blocks.csv",
        [
            "geometry", "geometry_description", "block_index", "blackout_angle_deg",
            "acquisition_seed", "acquisition_coordinate_sha256",
            "contact_noise_draw_sha256", "truth_sha256", "method",
            "replication_unit", "within_geometry_repeat",
            "horizon_rule", "horizon", "pseudo_dt", "terminal_window_states",
            "n_training", "n_masked_grid_points", "masked_score_rmse",
            "masked_score_mae", "masked_score_bias", "masked_scaled_score_mse",
            "masked_calibration_intercept", "masked_calibration_slope",
            "masked_calibration_slope_abs_error", "masked_calibration_in_the_large",
            "masked_calibration_ece", "masked_score_correlation",
            "masked_out_of_range_fraction", "masked_dice", "masked_balanced_accuracy",
            "masked_sensitivity", "masked_specificity", "whole_score_rmse", "gap_count",
            "truth_gap_count", "gap_count_abs_error", "total_gap_width_mm",
            "truth_total_gap_width_mm", "total_gap_width_abs_error_mm",
            "largest_gap_width_mm", "truth_largest_gap_width_mm",
            "largest_gap_width_abs_error_mm", "normalized_capacity",
            "truth_normalized_capacity", "capacity_abs_error", *CAPACITY_COLUMNS,
            "score_min", "score_max", "reconstruction_n", "reconstruction_dx_mm",
            "fv_n", "subcells_per_axis", "blackout_arc_width_mm", "blackout_buffer_mm",
            "zero_confidence_fraction", "holdout_confidence_max",
            "holdout_data_forcing_max_abs", "noise_rms", "screen_relative_residual",
            *ensemble_nan,
        ],
        allowed_nan=ensemble_nan,
    )
    geometry_names = {
        "complete_ring", "narrow_gap", "wide_gap", "two_gaps", "oblique_gap"
    }
    geometry_methods = {"screened", "passive", "graph"}
    geometry_angles = {0.0, 90.0, 180.0, 270.0}
    geometry_seeds = {3101, 3103, 3107, 3109}
    require(len(geometry_blocks) == 60, "geometry stress test must contain 60 block rows")
    require(set(geometry_blocks["geometry"]) == geometry_names, "geometry stress test has the wrong lesion scenarios")
    require(set(geometry_blocks["method"]) == geometry_methods, "geometry stress test has an incomplete method set")
    require(set(geometry_blocks["blackout_angle_deg"].astype(float)) == geometry_angles, "geometry stress test has the wrong fixed block rotations")
    require(set(geometry_blocks["acquisition_seed"].astype(int)) == geometry_seeds, "geometry stress test has the wrong acquisition seeds")
    for column in [
        "acquisition_coordinate_sha256", "contact_noise_draw_sha256", "truth_sha256"
    ]:
        require(bool(geometry_blocks[column].str.fullmatch(r"[0-9a-f]{64}").all()), f"geometry {column} is not a SHA-256 digest")
    expected_block_design = {
        1: (0.0, 3101), 2: (90.0, 3103), 3: (180.0, 3107), 4: (270.0, 3109)
    }
    for block_index, (angle, seed) in expected_block_design.items():
        subset = geometry_blocks.loc[geometry_blocks["block_index"] == block_index]
        require(bool((subset["blackout_angle_deg"] == angle).all()) and bool((subset["acquisition_seed"] == seed).all()), f"geometry block {block_index} has the wrong angle/seed pairing")
        require(subset["acquisition_coordinate_sha256"].nunique() == 1 and subset["contact_noise_draw_sha256"].nunique() == 1, f"geometry block {block_index} does not reuse its acquisition across geometries")
    require(not geometry_blocks.duplicated(["geometry", "block_index", "method"]).any(), "geometry stress test contains duplicate rows")
    require(
        geometry_blocks.groupby(["geometry", "block_index"])["method"].apply(set).map(
            lambda values: values == geometry_methods
        ).all(),
        "a geometry/block combination lacks one reconstruction method",
    )
    require(bool((geometry_blocks["replication_unit"] == "geometry").all()), "geometry is not declared as the replication unit")
    require(bool((geometry_blocks["within_geometry_repeat"] == "fixed_spatial_block").all()), "within-geometry repeats are not fixed spatial blocks")
    require(bool((geometry_blocks["horizon_rule"] == "fixed_common_120_step_budget_no_selection").all()), "geometry horizon rule is not the fixed computational budget")
    dynamic_geometry = geometry_blocks["method"].isin(["passive", "graph"])
    require(bool(np.allclose(geometry_blocks.loc[dynamic_geometry, "horizon"], 1.20)), "passive and graph geometry runs do not share T=1.20")
    require(bool((geometry_blocks.loc[~dynamic_geometry, "horizon"] == 0.0).all()), "screened geometry rows have nonzero pseudo-time")
    require(bool((geometry_blocks.loc[dynamic_geometry, "terminal_window_states"] == 12).all()), "geometry dynamic rows use the wrong terminal window")
    require(bool((geometry_blocks.loc[~dynamic_geometry, "terminal_window_states"] == 0).all()), "screened geometry rows use terminal averaging")
    require(bool(np.allclose(geometry_blocks["pseudo_dt"], 0.01)), "geometry stress test uses the wrong pseudo-time step")
    require(bool((geometry_blocks["n_masked_grid_points"] > 0).all()), "a spatial validation block is empty")
    require(bool((geometry_blocks["holdout_confidence_max"] == 0.0).all()), "geometry confidence is not exactly zero in a validation block")
    require(bool((geometry_blocks["holdout_data_forcing_max_abs"] == 0.0).all()), "geometry forcing is not exactly zero in a validation block")
    require(float(geometry_blocks["screen_relative_residual"].max()) < 2.0e-9, "geometry screened initialization is inaccurate")
    require(bool(np.allclose(geometry_blocks["masked_scaled_score_mse"], 0.25 * geometry_blocks["masked_score_rmse"] ** 2, rtol=2.0e-12, atol=2.0e-14)), "scaled-score MSE is inconsistent with score RMSE")
    require(bool((geometry_blocks["masked_calibration_slope_abs_error"] >= 0.0).all()), "geometry calibration-slope error is negative")
    for column in [
        "masked_calibration_ece", "masked_out_of_range_fraction", "masked_dice", "masked_balanced_accuracy",
        "masked_sensitivity", "masked_specificity",
    ]:
        require(bool(((geometry_blocks[column] >= 0.0) & (geometry_blocks[column] <= 1.0 + 1.0e-12)).all()), f"geometry {column} lies outside its admissible range")
    require(bool(np.allclose(geometry_blocks["gap_count_abs_error"], np.abs(geometry_blocks["gap_count"] - geometry_blocks["truth_gap_count"]))), "geometry gap-count errors are inconsistent")
    require(bool(np.allclose(geometry_blocks["total_gap_width_abs_error_mm"], np.abs(geometry_blocks["total_gap_width_mm"] - geometry_blocks["truth_total_gap_width_mm"]))), "geometry total-gap-width errors are inconsistent")
    require(bool(np.allclose(geometry_blocks["largest_gap_width_abs_error_mm"], np.abs(geometry_blocks["largest_gap_width_mm"] - geometry_blocks["truth_largest_gap_width_mm"]))), "geometry largest-gap-width errors are inconsistent")
    require(bool(np.allclose(geometry_blocks["capacity_abs_error"], np.abs(geometry_blocks["normalized_capacity"] - geometry_blocks["truth_normalized_capacity"]))), "geometry capacity errors are inconsistent")
    expected_truth_counts = {
        "complete_ring": 0, "narrow_gap": 1, "wide_gap": 1,
        "two_gaps": 2, "oblique_gap": 1,
    }
    for geometry, expected in expected_truth_counts.items():
        subset = geometry_blocks.loc[geometry_blocks["geometry"] == geometry]
        require(bool((subset["truth_gap_count"] == expected).all()), f"{geometry}: hidden gap count is wrong")
    invariant_columns = [
        "blackout_angle_deg", "acquisition_seed", "n_training", "n_masked_grid_points",
        "zero_confidence_fraction", "screen_relative_residual",
        "acquisition_coordinate_sha256", "contact_noise_draw_sha256", "truth_sha256",
    ]
    require(bool((geometry_blocks.groupby(["geometry", "block_index"])[invariant_columns].nunique(dropna=False) == 1).all().all()), "methods do not share a geometry/block acquisition and initialization")
    truth_columns = [
        "truth_gap_count", "truth_total_gap_width_mm", "truth_largest_gap_width_mm",
        "truth_normalized_capacity", "truth_sha256",
    ]
    require(bool((geometry_blocks.groupby("geometry")[truth_columns].nunique(dropna=False) == 1).all().all()), "a hidden geometry changes across blocks or methods")
    validate_capacity_identity(geometry_blocks, "geometry-replicated reconstruction")
    geometry_graph = geometry_blocks.loc[geometry_blocks["method"] == "graph"]
    for column, tolerance in {
        "max_state_residual": 1.0e-6, "max_primal_residual": 1.0e-6,
        "max_dual_residual": 1.0e-6, "max_box_residual": 1.0e-10,
        "max_complementarity_residual": 2.0e-6,
        "max_graph_projection_residual": 1.0e-6, "max_mass_defect": 1.0e-12,
    }.items():
        require(float(geometry_graph[column].max()) < tolerance, f"geometry graph flow: {column} exceeds tolerance")

    geometry_units = load_applied_csv(
        "sparse_reconstruction_geometry_units.csv",
        [
            "geometry", "geometry_description", "method", "replication_unit",
            "n_spatial_blocks_averaged", "block_averaging_rule", "horizon_rule",
            "horizon", "masked_score_rmse", "masked_score_mae", "masked_score_bias",
            "masked_scaled_score_mse", "masked_dice", "masked_balanced_accuracy",
            "whole_score_rmse", "capacity_abs_error", "gap_count_abs_error",
            "total_gap_width_abs_error_mm", "largest_gap_width_abs_error_mm",
            "masked_score_rmse_within_block_sd", "masked_calibration_intercept",
            "masked_calibration_slope", "masked_calibration_slope_abs_error",
            "masked_calibration_in_the_large", "masked_calibration_ece",
            "masked_score_correlation", "masked_out_of_range_fraction",
        ],
    )
    require(len(geometry_units) == 15 and not geometry_units.duplicated(["geometry", "method"]).any(), "geometry-level table must contain one row per geometry and method")
    require(set(geometry_units["geometry"]) == geometry_names and set(geometry_units["method"]) == geometry_methods, "geometry-level table has incomplete factors")
    require(bool((geometry_units["replication_unit"] == "geometry").all()), "geometry-level table changes the replication unit")
    require(bool((geometry_units["n_spatial_blocks_averaged"] == 4).all()), "geometry-level results do not average four blocks")
    require(bool((geometry_units["block_averaging_rule"] == "arithmetic_mean_before_method_contrast").all()), "geometry-level block averaging is not documented")
    averaged_columns = [
        "masked_score_rmse", "masked_score_mae", "masked_score_bias",
        "masked_scaled_score_mse", "masked_dice", "masked_balanced_accuracy",
        "whole_score_rmse", "capacity_abs_error", "gap_count_abs_error",
        "total_gap_width_abs_error_mm", "largest_gap_width_abs_error_mm",
    ]
    expected_units = geometry_blocks.groupby(["geometry", "method"])[averaged_columns].mean()
    for row in geometry_units.itertuples(index=False):
        expected = expected_units.loc[(row.geometry, row.method)]
        for column in averaged_columns:
            require(bool(np.isclose(float(getattr(row, column)), float(expected[column]), rtol=2.0e-12, atol=2.0e-14)), f"{row.geometry}/{row.method}: geometry-level {column} is not the four-block mean")

    geometry_summary = load_applied_csv(
        "sparse_reconstruction_geometry_summary.csv",
        [
            "metric", "n_geometries", "screened_mean", "passive_mean", "graph_mean",
            "graph_minus_passive_mean", "graph_minus_passive_median",
            "graph_minus_passive_min", "graph_minus_passive_max",
            "graph_better_geometry_count", "tied_geometry_count",
            "graph_worse_geometry_count", "interval_type",
        ],
    )
    summary_metrics = {
        "masked_score_rmse", "masked_scaled_score_mse", "masked_calibration_ece",
        "masked_calibration_slope_abs_error", "capacity_abs_error",
        "gap_count_abs_error", "largest_gap_width_abs_error_mm",
    }
    require(len(geometry_summary) == len(summary_metrics) and set(geometry_summary["metric"]) == summary_metrics, "geometry summary has the wrong metric set")
    require(bool((geometry_summary["n_geometries"] == 5).all()), "geometry summary uses the wrong number of units")
    require(bool((geometry_summary["interval_type"] == "none_fixed_prespecified_scenario_set").all()), "geometry summary makes an unsupported population interval")
    require(bool((geometry_summary[["graph_better_geometry_count", "tied_geometry_count", "graph_worse_geometry_count"]].sum(axis=1) == 5).all()), "geometry comparison counts do not sum to five")
    for row in geometry_summary.itertuples(index=False):
        pivot = geometry_units.pivot(index="geometry", columns="method", values=row.metric)
        difference = (pivot["graph"] - pivot["passive"]).to_numpy(dtype=float)
        expected = {
            "screened_mean": float(pivot["screened"].mean()),
            "passive_mean": float(pivot["passive"].mean()),
            "graph_mean": float(pivot["graph"].mean()),
            "graph_minus_passive_mean": float(np.mean(difference)),
            "graph_minus_passive_median": float(np.median(difference)),
            "graph_minus_passive_min": float(np.min(difference)),
            "graph_minus_passive_max": float(np.max(difference)),
            "graph_better_geometry_count": int(np.sum(difference < -1.0e-14)),
            "tied_geometry_count": int(np.sum(np.abs(difference) <= 1.0e-14)),
            "graph_worse_geometry_count": int(np.sum(difference > 1.0e-14)),
        }
        for column, value in expected.items():
            require(bool(np.isclose(float(getattr(row, column)), float(value), rtol=2.0e-12, atol=2.0e-14)), f"geometry summary has the wrong {column} for {row.metric}")

    calibration = load_applied_csv(
        "sparse_reconstruction_geometry_calibration.csv",
        [
            "scope", "geometry", "method", "bin", "bin_lower", "bin_upper",
            "n", "mean_predicted", "mean_truth",
        ],
        allowed_nan={"mean_predicted", "mean_truth"},
    )
    require(len(calibration) == 180, "geometry reliability table must contain 180 fixed-bin rows")
    require(set(calibration["scope"]) == {"geometry", "pooled_descriptive"}, "geometry reliability table has the wrong scopes")
    require(set(calibration["method"]) == geometry_methods, "geometry reliability table has incomplete methods")
    require(bool(((calibration["bin"] >= 1) & (calibration["bin"] <= 10)).all()), "geometry reliability bins are invalid")
    require(bool(np.allclose(calibration["bin_lower"], (calibration["bin"] - 1) / 10.0)) and bool(np.allclose(calibration["bin_upper"], calibration["bin"] / 10.0)), "geometry reliability bin edges are inconsistent")
    nonempty = calibration["n"] > 0
    require(bool(calibration.loc[nonempty, ["mean_predicted", "mean_truth"]].notna().all().all()), "nonempty reliability bins omit means")
    require(bool(calibration.loc[~nonempty, ["mean_predicted", "mean_truth"]].isna().all().all()), "empty reliability bins contain means")
    expected_masked_points = geometry_blocks.groupby(["geometry", "method"])["n_masked_grid_points"].sum()
    geometry_calibration = calibration.loc[calibration["scope"] == "geometry"]
    observed_points = geometry_calibration.groupby(["geometry", "method"])["n"].sum()
    require(bool((observed_points == expected_masked_points).all()), "geometry reliability bins do not contain every masked score")

    geometry_maps_path = DATA / "sparse_reconstruction_geometry_maps.npz"
    require(geometry_maps_path.exists(), "missing geometry reconstruction maps")
    geometry_maps = np.load(geometry_maps_path)
    required_geometry_maps = {
        *(f"truth_{name}" for name in geometry_names),
        "representative_truth", "representative_confidence",
        "representative_holdout_mask", "representative_train_x",
        "representative_train_y", "representative_screened",
        "representative_passive", "representative_graph",
    }
    require(required_geometry_maps.issubset(geometry_maps.files), "geometry map archive is incomplete")
    require(bool(np.all(geometry_maps["representative_confidence"][geometry_maps["representative_holdout_mask"].astype(bool)] == 0.0)), "representative geometry map has nonzero held-out confidence")


def validate_pvi_outputs() -> None:
    phase = load_applied_csv(
        "pvi_capacity_phase_diagram.csv",
        [
            "n", "dx_mm", "gap_width_mm", "gap_diffusivity_fraction",
            "nominal_gap_diffusivity_fraction", "realised_centerline_gap_diffusivity_fraction",
            "gap_angle_deg", "gap_orientation_relative_to_fiber_deg", "fiber_axis", "face_average",
            "lesion_width_mm", "transition_scale_mm", "d_long_mm2_per_ms", "d_trans_mm2_per_ms",
            "cells_across_zero_level_gap", "cells_across_transition", *CAPACITY_COLUMNS,
            "subcells_per_axis",
        ],
    )
    require(phase["gap_angle_deg"].nunique() >= 3, "capacity diagram contains too few gap orientations")
    require(phase["gap_width_mm"].nunique() >= 4, "capacity diagram contains too few gap widths")
    require(phase["nominal_gap_diffusivity_fraction"].nunique() >= 5, "capacity diagram contains too few gap diffusivities")
    require(set(phase["face_average"]) == {"harmonic"}, "capacity diagram must use harmonic face conductivities")
    require(bool((phase["subcells_per_axis"] >= 3).all()), "capacity diagram lacks subcell coefficient integration")
    require(float(phase["cells_across_zero_level_gap"].min()) >= 4.0, "capacity diagram contains an under-resolved zero-level gap")
    require(float(phase["cells_across_transition"].min()) >= 3.0, "capacity diagram contains an under-resolved interface")
    validate_capacity_identity(phase, "PVI capacity diagram")

    widths = tuple(sorted(phase["gap_width_mm"].unique()))
    diffusivities = tuple(sorted(phase["nominal_gap_diffusivity_fraction"].unique()))
    for angle, angle_group in phase.groupby("gap_angle_deg"):
        require(len(angle_group) == len(widths) * len(diffusivities), f"angle {angle:g}: incomplete width-diffusivity capacity grid")
        require(not angle_group.duplicated(["gap_width_mm", "nominal_gap_diffusivity_fraction"]).any(), f"angle {angle:g}: duplicated capacity point")
        for nominal, group in angle_group.groupby("nominal_gap_diffusivity_fraction"):
            ordered = group.sort_values("gap_width_mm")
            require(tuple(ordered["gap_width_mm"]) == widths, f"angle {angle:g}, eta {nominal:g}: missing width")
            require(bool(np.all(np.diff(ordered["normalized_capacity"]) > 0.0)), f"angle {angle:g}, eta {nominal:g}: capacity is not strictly increasing in width")
        for width, group in angle_group.groupby("gap_width_mm"):
            ordered = group.sort_values("nominal_gap_diffusivity_fraction")
            require(tuple(ordered["nominal_gap_diffusivity_fraction"]) == diffusivities, f"angle {angle:g}, width {width:g}: missing diffusivity")
            require(bool(np.all(np.diff(ordered["normalized_capacity"]) > 0.0)), f"angle {angle:g}, width {width:g}: capacity is not strictly increasing in nominal diffusivity")

    ep_nan = {"matched_case", "matched_pair_relative_capacity_difference", "crossing_time_ms", "sector_first_arrival_ms", "probe_activation_time_ms"}
    ep = load_applied_csv(
        "pvi_bidirectional_ep.csv",
        [
            "experiment", "case_id", "matched_case", "matched_pair_relative_capacity_difference",
            "face_average", "n", "dx_mm", "dt_ms", "gap_width_mm",
            "gap_diffusivity_fraction", "gap_angle_deg", "pacing_direction", *CAPACITY_COLUMNS,
            "crossing_time_ms", "sector_first_arrival_ms", "probe_activation_time_ms",
            "crossed_by_horizon", "captured_by_horizon", "target_activated_fraction",
            "sector_activated_fraction", "capture_fraction_threshold", "target_cell_count",
            "target_peak_voltage", "probe_peak_voltage", "t_end_ms", "target_sector_half_angle_deg",
            "cfl", "max_diffusion_mass_defect", "nominal_gap_diffusivity_fraction",
            "realised_centerline_gap_diffusivity_fraction", "subcells_per_axis",
        ],
        ep_nan,
    )
    required_experiments = {"stratified", "complete_ring_control", "matched_capacity", "face_mean_sensitivity"}
    require(required_experiments.issubset(set(ep["experiment"])), "bidirectional EP table omits a required control or sensitivity experiment")
    require_binary(ep, ["crossed_by_horizon", "captured_by_horizon"], "bidirectional EP")
    require(bool(np.allclose(ep["capture_fraction_threshold"], 0.8)), "bidirectional EP does not document the 80% capture endpoint")
    expected_capture = ep["sector_activated_fraction"] >= ep["capture_fraction_threshold"]
    require(bool((ep["captured_by_horizon"].astype(bool) == expected_capture).all()), "saved capture label is inconsistent with the 80% sector endpoint")
    require(bool((ep["crossed_by_horizon"] >= ep["captured_by_horizon"]).all()), "capture is reported without a distal-sector arrival")
    crossed = ep["crossed_by_horizon"].astype(bool)
    require(bool(ep.loc[crossed, ["crossing_time_ms", "sector_first_arrival_ms", "probe_activation_time_ms"]].notna().all().all()), "a crossing row lacks first-arrival or fixed-probe timing")
    require(bool(ep.loc[~crossed, ["crossing_time_ms", "sector_first_arrival_ms", "probe_activation_time_ms"]].isna().all().all()), "a censored non-crossing row stores an activation time")
    require(bool(np.allclose(ep.loc[crossed, "crossing_time_ms"], ep.loc[crossed, "sector_first_arrival_ms"])), "crossing time and distal-sector first arrival disagree")
    require(bool((ep["target_cell_count"] > 0).all()), "EP target sector has no cells")
    require(float(ep["cfl"].max()) < 0.95, "bidirectional EP violates the explicit CFL bound")
    require(float(ep["max_diffusion_mass_defect"].max()) < 1.0e-12, "bidirectional EP diffusion update is not conservative")
    validate_capacity_identity(ep, "bidirectional EP")

    stratified = ep.loc[ep["experiment"] == "stratified"]
    require(set(stratified["pacing_direction"]) == {"exit", "entrance"}, "stratified cases are not paced bidirectionally")
    require(stratified.groupby("case_id")["pacing_direction"].apply(set).map(lambda values: values == {"exit", "entrance"}).all(), "a stratified case is missing one pacing direction")
    for direction, group in stratified.groupby("pacing_direction"):
        require(group["captured_by_horizon"].nunique() == 2, f"{direction}: stratified EP contains no conducting-to-blocked transition")

    control = ep.loc[ep["experiment"] == "complete_ring_control"]
    require(set(control["pacing_direction"]) == {"exit", "entrance"}, "complete-ring negative control is not bidirectional")
    require(bool((control["gap_width_mm"] == 0.0).all()), "complete-ring control has a nonzero gap")
    require(bool((control["captured_by_horizon"] == 0).all()), "complete-ring negative control shows capture")
    require(bool((control["sector_activated_fraction"] < control["capture_fraction_threshold"]).all()), "complete-ring control reaches the capture threshold")

    matched = ep.loc[ep["experiment"] == "matched_capacity"]
    require(matched["matched_case"].nunique() == 2, "matched-capacity experiment needs two cases")
    require(matched.groupby("matched_case")["pacing_direction"].apply(set).map(lambda values: values == {"exit", "entrance"}).all(), "matched-capacity pair is not paced bidirectionally")
    case_capacity = matched.groupby("matched_case")["normalized_capacity"].first()
    pair_difference = abs(float(case_capacity.iloc[0] - case_capacity.iloc[1])) / float(case_capacity.mean())
    require(pair_difference < 0.02, "matched pair differs by more than 2% in normalized capacity")
    require(bool(np.allclose(matched["matched_pair_relative_capacity_difference"], pair_difference)), "saved matched-capacity difference is inconsistent")
    outcome_table = matched.pivot(index="pacing_direction", columns="matched_case", values="captured_by_horizon")
    require(bool((outcome_table.nunique(axis=1) > 1).any()), "matched-capacity cases do not produce a contrasting EP outcome")

    face = ep.loc[ep["experiment"] == "face_mean_sensitivity"]
    require(set(face["face_average"]) == {"harmonic", "arithmetic"}, "face-mean sensitivity lacks harmonic or arithmetic averaging")
    comparison_columns = ["n", "dt_ms", "gap_width_mm", "gap_diffusivity_fraction", "gap_angle_deg", "pacing_direction"]
    require(all(face[column].nunique() == 1 for column in comparison_columns), "face-mean sensitivity changes more than the interface average")
    require(face["normalized_capacity"].nunique() == 2, "harmonic and arithmetic face means produced identical saved capacities")

    representative_path = DATA / "pvi_capacity_representatives.npz"
    require(representative_path.exists(), "missing PVI capacity representative maps")
    representative = np.load(representative_path)
    required_maps = {"diffusivity", "potential", "exit_activation", "exit_peak", "entrance_activation", "entrance_peak"}
    require(required_maps.issubset(representative.files), f"PVI representative lacks {sorted(required_maps.difference(representative.files))}")
    shapes = {representative[key].shape for key in required_maps}
    require(len(shapes) == 1, "PVI representative arrays have inconsistent shapes")
    for key in ["diffusivity", "potential", "exit_peak", "entrance_peak"]:
        require(bool(np.isfinite(representative[key]).all()), f"PVI representative {key} contains a non-finite value")
    for key in ["exit_activation", "entrance_activation"]:
        require(not bool(np.isinf(representative[key]).any()) and bool(np.isfinite(representative[key]).any()), f"PVI representative {key} has invalid censoring")

    refinement_nan = {
        "dt_ms", "crossing_time_ms", "crossed_by_horizon", "probe_activation_time_ms",
        "sector_activated_fraction", "t_end_ms", "face_average", "target_peak_voltage", "cfl",
        "matched_case", "matched_pair_relative_capacity_difference", "pacing_direction",
        "gap_width_mm", "gap_diffusivity_fraction", "gap_angle_deg", "captured_by_horizon",
    }
    refinement = load_applied_csv(
        "pvi_capacity_refinement.csv",
        [
            "quantity", "regime", "n", "dx_mm", "dt_ms", "subcells_per_axis", *CAPACITY_COLUMNS,
            "crossing_time_ms", "crossed_by_horizon", "probe_activation_time_ms",
            "sector_activated_fraction", "t_end_ms", "face_average", "target_peak_voltage", "cfl",
            "matched_case", "matched_pair_relative_capacity_difference", "pacing_direction",
            "gap_width_mm", "gap_diffusivity_fraction", "gap_angle_deg", "captured_by_horizon",
        ],
        refinement_nan,
    )
    validate_capacity_identity(refinement, "PVI refinement")
    capacity_refinement = refinement.loc[refinement["quantity"] == "capacity"].sort_values("n")
    require(capacity_refinement["n"].nunique() >= 4, "capacity refinement has too few grids")
    require(bool(capacity_refinement[["dt_ms", "crossing_time_ms", "crossed_by_horizon", "cfl"]].isna().all().all()), "elliptic capacity rows contain EP metadata")
    require(bool(np.all(np.diff(capacity_refinement["dx_mm"]) < 0.0)), "capacity dx does not decrease with n")
    ep_space = refinement.loc[refinement["quantity"] == "EP"]
    for regime, group in ep_space.groupby("regime"):
        require(group["n"].nunique() >= 3 and group["dt_ms"].nunique() == 1, f"{regime}: EP spatial refinement is not independent")
    require({"conducting", "low_conductance"}.issubset(set(ep_space["regime"])), "EP spatial refinement lacks conducting or low-conductance regime")
    ep_time = refinement.loc[refinement["quantity"] == "EP_dt"]
    require(ep_time["dt_ms"].nunique() >= 3 and ep_time["n"].nunique() == 1, "EP time refinement is not independent")
    matched_space = refinement.loc[refinement["quantity"] == "matched_EP_space"]
    require(matched_space["n"].nunique() >= 3 and matched_space["dt_ms"].nunique() == 1,
            "matched pair lacks independent spatial refinement")
    require(set(matched_space["matched_case"]) == {"A", "B"},
            "matched spatial refinement lost one case")
    require(set(matched_space["pacing_direction"]) == {"exit", "entrance"},
            "matched spatial refinement is not bidirectional")
    require(matched_space.groupby(["n", "matched_case"])["pacing_direction"].apply(set).map(
        lambda values: values == {"exit", "entrance"}
    ).all(), "a matched spatial grid lacks one pacing direction")
    require(float(matched_space["matched_pair_relative_capacity_difference"].max()) < 0.005,
            "matched capacities separate by more than 0.5% under spatial refinement")
    matched_exit = matched_space.loc[matched_space["pacing_direction"] == "exit"].pivot(
        index="n", columns="matched_case", values="captured_by_horizon"
    )
    require(bool(((matched_exit["A"] == 1.0) & (matched_exit["B"] == 0.0)).all()),
            "matched-pair exit-pacing contrast is not stable across grids")

    matched_time = refinement.loc[refinement["quantity"] == "matched_EP_time"]
    require(matched_time["dt_ms"].nunique() >= 3 and matched_time["n"].nunique() == 1,
            "matched pair lacks independent time-step refinement")
    require(set(matched_time["matched_case"]) == {"A", "B"}
            and set(matched_time["pacing_direction"]) == {"exit", "entrance"},
            "matched time refinement is incomplete")
    time_labels = matched_time.pivot_table(
        index="dt_ms", columns=["matched_case", "pacing_direction"],
        values="captured_by_horizon", aggfunc="first"
    )
    require(bool((time_labels[("A", "exit")] == 1.0).all()
                 and (time_labels[("A", "entrance")] == 1.0).all()
                 and (time_labels[("B", "exit")] == 0.0).all()
                 and (time_labels[("B", "entrance")] == 0.0).all()),
            "matched-pair production-grid labels change across tested time steps")

    dynamic = refinement.loc[refinement["quantity"].isin(
        ["EP", "EP_dt", "matched_EP_space", "matched_EP_time"]
    )]
    require(bool(dynamic[["dt_ms", "crossed_by_horizon", "captured_by_horizon", "sector_activated_fraction", "t_end_ms", "face_average", "target_peak_voltage", "cfl"]].notna().all().all()), "EP refinement omits required endpoint metadata")
    require_binary(dynamic, ["crossed_by_horizon", "captured_by_horizon"], "PVI refinement")
    expected_dynamic_capture = dynamic["sector_activated_fraction"] >= 0.8
    require(bool((dynamic["captured_by_horizon"].astype(bool) == expected_dynamic_capture).all()),
            "PVI refinement capture label is inconsistent with the 80% sector endpoint")
    require(float(dynamic["cfl"].max()) < 0.95, "PVI refinement violates the explicit CFL bound")


# RETIRED LEGACY VALIDATOR: not called by the publication entry point.
def _validate_retired_synthetic_end_to_end_outputs() -> None:
    predicted_endpoint_columns = [
        "exit_crossing_time_ms", "exit_crossed_by_horizon", "exit_peak_voltage",
        "exit_probe_activation_time_ms", "exit_sector_activated_fraction",
        "entrance_crossing_time_ms", "entrance_crossed_by_horizon", "entrance_peak_voltage",
        "entrance_probe_activation_time_ms", "entrance_sector_activated_fraction",
        "exit_crossing_class_correct", "exit_crossing_time_error_ms", "exit_peak_voltage_error",
        "entrance_crossing_class_correct", "entrance_crossing_time_error_ms", "entrance_peak_voltage_error",
    ]
    ensemble = load_applied_csv(
        "end_to_end_ensemble.csv",
        [
            "seed", "method", "ep_included", "ep_n", "ep_dx_mm", "ep_dt_ms",
            "subcells_per_axis", "face_average", *CAPACITY_COLUMNS, "truth_normalized_capacity",
            "capacity_error", "capacity_abs_error", "rmse", *predicted_endpoint_columns,
            "capture_fraction_threshold", "truth_exit_crossing_time_ms",
            "truth_exit_crossed_by_horizon", "truth_exit_peak_voltage",
            "truth_exit_probe_activation_time_ms", "truth_exit_sector_activated_fraction",
            "truth_entrance_crossing_time_ms", "truth_entrance_crossed_by_horizon",
            "truth_entrance_peak_voltage", "truth_entrance_probe_activation_time_ms",
            "truth_entrance_sector_activated_fraction",
        ],
        predicted_endpoint_columns,
    )
    require(set(ensemble["method"]) == {"screened", "passive", "graph"}, "end-to-end ensemble has an incomplete method set")
    require(ensemble.groupby("seed")["method"].apply(set).map(lambda values: values == {"screened", "passive", "graph"}).all(), "end-to-end seeds are not paired across methods")
    require_binary(ensemble, ["ep_included", "truth_exit_crossed_by_horizon", "truth_entrance_crossed_by_horizon"], "end-to-end ensemble")
    require(bool(np.allclose(ensemble["capture_fraction_threshold"], 0.8)), "end-to-end table does not document the 80% capture endpoint")
    require(bool(np.allclose(ensemble["capacity_error"], ensemble["normalized_capacity"] - ensemble["truth_normalized_capacity"])), "signed end-to-end capacity error is inconsistent")
    require(bool(np.allclose(ensemble["capacity_abs_error"], np.abs(ensemble["capacity_error"]))), "absolute end-to-end capacity error is inconsistent")
    validate_capacity_identity(ensemble, "end-to-end ensemble")

    truth_columns = [column for column in ensemble.columns if column.startswith("truth_")]
    for column in truth_columns:
        require(ensemble[column].nunique(dropna=False) == 1, f"end-to-end truth field {column} changes across reconstructions")
    included = ensemble["ep_included"].astype(bool)
    require(bool(ensemble.loc[~included, predicted_endpoint_columns].isna().all().all()), "rows excluded from EP contain predicted EP outputs")
    require(bool(ensemble.loc[included, ["exit_crossed_by_horizon", "exit_peak_voltage", "exit_sector_activated_fraction", "entrance_crossed_by_horizon", "entrance_peak_voltage", "entrance_sector_activated_fraction", "exit_crossing_class_correct", "exit_peak_voltage_error", "entrance_crossing_class_correct", "entrance_peak_voltage_error"]].notna().all().all()), "included end-to-end EP rows omit endpoint diagnostics")
    require(set(ensemble.loc[included, "method"]) == {"passive", "graph"}, "end-to-end EP subset must contain paired passive and graph reconstructions")
    included_seeds = ensemble.loc[included].groupby("method")["seed"].apply(frozenset)
    require(included_seeds.nunique() == 1 and len(included_seeds.iloc[0]) >= 5, "passive and graph EP subsets are not paired")

    for side in ["exit", "entrance"]:
        pred_label = f"{side}_crossed_by_horizon"
        truth_label = f"truth_{side}_crossed_by_horizon"
        class_column = f"{side}_crossing_class_correct"
        require_binary(ensemble.loc[included], [pred_label, class_column], f"end-to-end {side} endpoint")
        expected_class = ensemble.loc[included, pred_label].astype(int) == ensemble.loc[included, truth_label].astype(int)
        require(bool((ensemble.loc[included, class_column].astype(bool) == expected_class).all()), f"{side}: crossing classification diagnostic is inconsistent")
        predicted_crossing = included & ensemble[pred_label].eq(1.0)
        require(bool(ensemble.loc[predicted_crossing, [f"{side}_crossing_time_ms", f"{side}_probe_activation_time_ms"]].notna().all().all()), f"{side}: predicted crossing lacks timing diagnostics")
        require(bool(ensemble.loc[included & ~ensemble[pred_label].eq(1.0), [f"{side}_crossing_time_ms", f"{side}_probe_activation_time_ms"]].isna().all().all()), f"{side}: censored crossing stores a time")
        both_cross = included & ensemble[pred_label].eq(1.0) & ensemble[truth_label].eq(1.0)
        expected_time_error = ensemble.loc[both_cross, f"{side}_crossing_time_ms"] - ensemble.loc[both_cross, f"truth_{side}_crossing_time_ms"]
        require(bool(np.allclose(ensemble.loc[both_cross, f"{side}_crossing_time_error_ms"], expected_time_error)), f"{side}: crossing-time error is inconsistent")
        expected_peak_error = ensemble.loc[included, f"{side}_peak_voltage"] - ensemble.loc[included, f"truth_{side}_peak_voltage"]
        require(bool(np.allclose(ensemble.loc[included, f"{side}_peak_voltage_error"], expected_peak_error)), f"{side}: peak-voltage error is inconsistent")

    fields_path = DATA / "end_to_end_fields.npz"
    require(fields_path.exists(), "missing end-to-end representative fields")
    fields = np.load(fields_path)
    required_fields = {
        "truth_diffusivity", "truth_potential", "truth_exit_activation", "truth_exit_peak",
        "truth_entrance_activation", "truth_entrance_peak", "graph_score", "graph_diffusivity",
        "graph_potential", "graph_exit_activation", "graph_exit_peak", "graph_entrance_activation",
        "graph_entrance_peak",
    }
    require(required_fields.issubset(fields.files), f"end-to-end fields lack {sorted(required_fields.difference(fields.files))}")
    ep_shape = fields["truth_diffusivity"].shape
    for key in required_fields.difference({"graph_score"}):
        require(fields[key].shape == ep_shape, f"end-to-end field {key} has the wrong shape")
        require(not bool(np.isinf(fields[key]).any()), f"end-to-end field {key} contains infinity")
    for key in ["truth_diffusivity", "truth_potential", "truth_exit_peak", "truth_entrance_peak", "graph_diffusivity", "graph_potential", "graph_exit_peak", "graph_entrance_peak", "graph_score"]:
        require(bool(np.isfinite(fields[key]).all()), f"end-to-end field {key} contains an undocumented NaN")
    for key in ["truth_exit_activation", "truth_entrance_activation", "graph_exit_activation", "graph_entrance_activation"]:
        require(bool(np.isfinite(fields[key]).any()), f"end-to-end activation field {key} contains no arrivals")
    require(bool((fields["truth_diffusivity"] > 0.0).all() and (fields["graph_diffusivity"] > 0.0).all()), "end-to-end diffusivity is not positive")

    refinement = load_applied_csv(
        "end_to_end_ep_refinement.csv",
        [
            "seed", "n", "dx_mm", "dt_ms", "normalized_capacity", "crossing_time_ms",
            "crossed_by_horizon", "target_peak_voltage", "cfl", "capacity_relative_residual",
            "capacity_flux_energy_defect",
        ],
    )
    require(refinement["n"].nunique() >= 3 and refinement["dt_ms"].nunique() == 1, "end-to-end EP grid refinement is not independent")
    require(bool(np.all(np.diff(refinement.sort_values("n")["dx_mm"]) < 0.0)), "end-to-end EP dx does not decrease with n")
    require(float(refinement["cfl"].max()) < 0.95, "end-to-end EP refinement violates the explicit CFL bound")
    require(float(refinement["capacity_relative_residual"].max()) < 1.0e-10, "end-to-end capacity solve residual is too large")
    require(float(refinement["capacity_flux_energy_defect"].max()) < 1.0e-9, "end-to-end capacity flux/energy defect is too large")

    summary = load_applied_csv(
        "end_to_end_summary.csv",
        ["method", "metric", "n", "mean", "sample_sd", "interval_type", "lower_95", "upper_95"],
    )
    require(bool((summary["n"] > 0).all()), "end-to-end summary has a non-positive sample count")
    require(bool((summary["lower_95"] <= summary["mean"]).all() and (summary["mean"] <= summary["upper_95"]).all()), "end-to-end confidence interval does not contain its estimate")
    for method in ["screened", "passive", "graph"]:
        direct = ensemble.loc[ensemble["method"] == method]
        for metric in ["rmse", "capacity_abs_error", "normalized_capacity"]:
            row = summary.loc[(summary["method"] == method) & (summary["metric"] == metric)]
            require(len(row) == 1, f"end-to-end summary lacks {method} {metric}")
            require(int(row["n"].iloc[0]) == len(direct), f"end-to-end summary has the wrong n for {method} {metric}")
            require(np.isclose(float(row["mean"].iloc[0]), float(direct[metric].mean())), f"end-to-end summary has the wrong mean for {method} {metric}")
    for method in ["passive", "graph"]:
        direct = ensemble.loc[included & ensemble["method"].eq(method)]
        for side in ["exit", "entrance"]:
            metric = f"{side}_crossing_fraction"
            row = summary.loc[(summary["method"] == method) & (summary["metric"] == metric)]
            require(len(row) == 1 and row["interval_type"].iloc[0] == "Wilson_score", f"end-to-end summary lacks a Wilson interval for {method} {side}")
            require(int(row["n"].iloc[0]) == len(direct), f"Wilson interval has the wrong n for {method} {side}")
            require(np.isclose(float(row["mean"].iloc[0]), float(direct[f"{side}_crossed_by_horizon"].mean())), f"Wilson interval has the wrong estimate for {method} {side}")
            require(0.0 <= float(row["lower_95"].iloc[0]) <= float(row["upper_95"].iloc[0]) <= 1.0, f"Wilson interval is outside [0,1] for {method} {side}")


def validate_figure_inventory() -> None:
    # The revision retains the six original figures and adds further figures.
    # Resolve the original inventory explicitly: a prefix count would reject
    # valid additions while saying nothing about the required original files.
    stems = (
        "fig1_exact_verification", "fig2_graph_limit",
        "fig3_conduction_calibration", "fig4_sparse_reconstruction",
        "fig5_pvi_capacity_phase_diagram", "fig6_patient_surface",
    )
    for stem in stems:
        pdf, png = FIGURES / (stem + ".pdf"), FIGURES / (stem + ".png")
        require(pdf.is_file() and png.is_file(), f"{stem} needs a PDF and PNG")
        require(pdf.stat().st_size > 10_000 and png.stat().st_size > 10_000,
                f"{stem} is unexpectedly small")


def validate_run_metadata() -> None:
    path = DATA / "run_metadata.json"
    require(path.is_file(), "missing run_metadata.json; run --refresh-run-metadata after all six experiments")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    require(metadata.get("schema_version") == 1, "unsupported run-metadata schema")
    require(metadata.get("mode") == "publication", "outputs were last written in quick mode")
    require(bool(metadata.get("environment")), "run metadata lacks environment details")
    require(bool(metadata.get("design")), "run metadata lacks design details")
    design = metadata["design"]
    require(
        design.get("publication_experiments") == [1, 2, 3, 4, 5, 6],
        "run metadata does not describe the current six experiments",
    )
    require(
        np.isclose(float(design.get("geometry_reconstruction_horizon", np.nan)), 1.20)
        and design.get("geometry_horizon_rule")
        == "fixed_common_120_step_budget_no_selection",
        "run metadata lacks the fixed geometry-reconstruction budget",
    )
    require(
        set(np.asarray(design.get("geometry_blackout_angles_deg", []), dtype=float))
        == {0.0, 90.0, 180.0, 270.0}
        and design.get("geometry_replication_unit") == "geometry",
        "run metadata lacks the spatial-block geometry design",
    )
    artifacts = metadata.get("artifacts", {})
    require(bool(artifacts), "run metadata has no artifact checksums")
    require(
        design.get("geometry_contact_noise") == "uncensored_gaussian",
        "run metadata does not describe uncensored geometry-contact noise",
    )
    require(
        design.get("patient_experiment") == "zenodo_prior_pvi_surface_reconstruction"
        and design.get("patient_source_doi") == "10.5281/zenodo.10726677"
        and design.get("patient_replication_unit")
        == "patient_after_averaging_left_right_masks"
        and design.get("patient_ids") == ["P1", "P3", "P4", "P5", "P6", "P7"],
        "run metadata lacks the real prior-PVI patient design",
    )
    # The producer owns an explicit list, not a glob. Exact equality rejects
    # both omitted current artifacts and accidental tracking of retired data.
    from run_applied import PUBLICATION_ARTIFACT_PATHS

    require(
        set(artifacts) == set(PUBLICATION_ARTIFACT_PATHS),
        "run metadata differs from the declared publication artifact list",
    )
    for relative, record in artifacts.items():
        artifact = ROOT / relative
        require(artifact.is_file(), f"checksummed artifact is missing: {relative}")
        payload = artifact.read_bytes()
        require(len(payload) == int(record["bytes"]), f"artifact size changed: {relative}")
        digest = hashlib.sha256(payload).hexdigest()
        require(digest == record["sha256"], f"artifact checksum changed: {relative}")


def main() -> None:
    validate_exact_solutions()
    validate_graph_limit()
    validate_calibration()
    validate_applied_outputs()
    from validate_zenodo_pvi_outputs import validate as validate_patient_outputs

    validate_patient_outputs(DATA, FIGURES)
    validate_figure_inventory()
    validate_run_metadata()
    print("all numerical checks passed")


if __name__ == "__main__":
    main()
