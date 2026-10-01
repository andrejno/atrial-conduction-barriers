"""Independent consistency checks for the Zenodo prior-PVI surface experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import run_zenodo_pvi_reconstruction as design
except ImportError:  # pragma: no cover
    from . import run_zenodo_pvi_reconstruction as design


ROOT = Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        _require(np.isfinite(values).all(), f"{column} contains a non-finite value")


def _key_set(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple[str, ...]]:
    return {
        tuple(str(value) for value in row)
        for row in frame[list(columns)].itertuples(index=False, name=None)
    }


def validate(data_dir: Path, figure_dir: Path) -> None:
    paths = {
        "inventory": data_dir / "zenodo_pvi_mesh_inventory.csv",
        "boundary": data_dir / "zenodo_pvi_boundary_transfer.csv",
        "metrics": data_dir / "zenodo_pvi_reconstruction_metrics.csv",
        "capacity": data_dir / "zenodo_pvi_capacity.csv",
        "summary": data_dir / "zenodo_pvi_patient_summary.csv",
        "contrasts": data_dir / "zenodo_pvi_patient_contrasts.csv",
        "representative": data_dir / "zenodo_pvi_representative.npz",
        "provenance": data_dir / "zenodo_pvi_provenance.json",
        "schema": data_dir / "zenodo_pvi_output_schema.json",
        "contact_geometry": data_dir / "zenodo_pvi_contact_geometry.csv",
    }
    for label, path in paths.items():
        _require(path.is_file() and path.stat().st_size > 0, f"missing/empty {label}: {path}")
    for suffix in ("pdf", "png"):
        path = figure_dir / f"fig6_patient_surface.{suffix}"
        _require(path.is_file() and path.stat().st_size > 1000, f"missing/empty figure {path}")

    inventory = pd.read_csv(paths["inventory"])
    boundary = pd.read_csv(paths["boundary"])
    metrics = pd.read_csv(paths["metrics"])
    capacity = pd.read_csv(paths["capacity"])
    summary = pd.read_csv(paths["summary"])
    contrasts = pd.read_csv(paths["contrasts"])
    contact_geometry = pd.read_csv(paths["contact_geometry"])
    patients = set(design.PATIENT_IDS)
    methods = {"screened", "passive", "graph"}
    masks = set(design.MASK_GROUPS)

    _require(len(inventory) == 6, "mesh inventory must have six LA patients")
    _require(set(inventory["patient_id"]) == patients, "mesh inventory patient set differs")
    _require(design.EXCLUDED_RIGHT_ATRIUM not in set(inventory["patient_id"]), "P2 RA leaked")
    _require((inventory["chamber"] == "LA").all(), "inventory contains a non-LA case")
    _require((inventory["prior_pvi"] == 1).all(), "not every included case is marked prior-PVI")
    _require((inventory["connected_components"] == 1).all(), "a surface is disconnected")
    _require((inventory["boundary_components"] == 5).all(), "an LA lacks five boundaries")
    _require((inventory["nonmanifold_edges"] == 0).all(), "nonmanifold surface accepted")
    _require((inventory["orientation_conflicts"] == 0).all(), "orientation conflict accepted")
    _require((inventory["removed_duplicate_triangles"] == 0).all(), "unexpected duplicates")
    _require(int(inventory["removed_zero_area_triangles"].sum()) == 1, "expected only P4 zero-area removal")
    _require(
        int(inventory.loc[inventory["patient_id"] == "P4", "removed_zero_area_triangles"].iloc[0]) == 1,
        "the one zero-area removal must belong to P4",
    )
    _finite(
        inventory,
        ["total_area_mm2", "minimum_angle_degrees", "maximum_edge_ratio", "bi_min_mv", "bi_median_mv", "bi_max_mv"],
    )
    _require((inventory["bi_min_mv"] >= 0.0).all(), "negative bipolar voltage accepted")

    expected_boundary_keys = {
        (patient, label) for patient in design.PATIENT_IDS for label in design.BOUNDARY_TAGS
    }
    _require(len(boundary) == 30, "boundary table must have 30 patient-boundary rows")
    _require(
        _key_set(boundary, ("patient_id", "label")) == expected_boundary_keys,
        "boundary patient/label keys are incomplete",
    )
    for label, tag in design.BOUNDARY_TAGS.items():
        chosen = boundary[boundary["label"] == label]
        _require((chosen["official_tag"] == tag).all(), f"wrong official tag for {label}")
    _require((boundary["p95_distance_mm"] <= 0.25 + 1e-14).all(), "boundary p95 transfer too large")
    _require((boundary["maximum_distance_mm"] <= 0.30 + 1e-14).all(), "boundary max transfer too large")
    _require((boundary["nearest_tag_fraction"] >= 0.95 - 1e-14).all(), "boundary tag purity too low")

    expected_metric_keys = {
        (patient, mask, method)
        for patient in design.PATIENT_IDS
        for mask in design.MASK_GROUPS
        for method in methods
    }
    _require(len(metrics) == 36, "reconstruction table must have 36 rows")
    expected_contact_keys = {
        (patient, mask) for patient in design.PATIENT_IDS for mask in design.MASK_GROUPS
    }
    _require(
        len(contact_geometry) == 12
        and _key_set(contact_geometry, ("patient_id", "mask_name")) == expected_contact_keys,
        "contact geometry table must have the twelve patient-mask rows",
    )
    _finite(contact_geometry, list(contact_geometry.select_dtypes(include=np.number).columns))
    contact_counts = metrics.groupby(["patient_id", "mask"])["n_contacts"]
    _require((contact_counts.nunique() == 1).all(), "methods did not receive the same number of contacts")
    joined_contacts = contact_geometry.merge(
        contact_counts.first().reset_index(),
        left_on=["patient_id", "mask_name"], right_on=["patient_id", "mask"],
        suffixes=("_geometry", "_reconstruction"), validate="one_to_one",
    )
    _require(
        (joined_contacts["n_contacts_geometry"] == joined_contacts["n_contacts_reconstruction"]).all(),
        "contact geometry check used different contacts from the reconstruction",
    )
    _require(
        np.allclose(contact_geometry["kernel_support_mm"], design.KERNEL_SUPPORT_MM)
        and np.allclose(
            contact_geometry["kernel_weight_fraction_edge_distance_gt_r"],
            contact_geometry["kernel_weight_edge_distance_gt_r"] / contact_geometry["total_kernel_weight"],
        ),
        "contact geometry kernel radius or weight fraction is inconsistent",
    )
    _require(
        _key_set(metrics, ("patient_id", "mask", "method")) == expected_metric_keys,
        "reconstruction key grid is incomplete",
    )
    _finite(
        metrics,
        [
            "rmse_area_weighted",
            "mean_squared_error_area_weighted",
            "integrated_squared_error_mm2",
            "mae_area_weighted",
            "binned_score_calibration_error",
            "uncensored_state_out_of_range_fraction",
            "dice_voltage_le_0p1",
            "balanced_accuracy_voltage_le_0p1",
            "solver_relative_residual",
        ],
    )
    phase_rows = metrics[metrics["method"] != "screened"]
    _finite(phase_rows, ["maximum_mass_source_defect", "maximum_graph_projection_residual"])
    screened_rows = metrics[metrics["method"] == "screened"]
    _require(screened_rows["maximum_mass_source_defect"].isna().all(), "screened mass diagnostic must be N/A")
    _require(screened_rows["maximum_graph_projection_residual"].isna().all(), "screened graph diagnostic must be N/A")
    _require((metrics["n_contacts"] > 50).all(), "a mask has implausibly few contacts")
    _require((metrics["evaluation_nodes"] > 50).all(), "a mask has too few evaluation nodes")
    _require((metrics["evaluation_area_mm2"] > 0.0).all(), "a mask has no evaluation area")
    _require((metrics["observed_area_fraction"] > 0.25).all(), "a mask has inadequate observed support")
    _require((metrics["observed_area_fraction"] < 0.95).all(), "observations are not sparse")
    _require((metrics["blackout_confidence_max"] == 0.0).all(), "blackout confidence is not exact zero")
    _require((metrics["blackout_forcing_max_abs"] == 0.0).all(), "blackout forcing is not exact zero")
    _require(
        np.allclose(
            metrics["rmse_area_weighted"] ** 2,
            metrics["mean_squared_error_area_weighted"],
            rtol=2e-12,
            atol=2e-14,
        ),
        "RMSE and area-weighted MSE disagree",
    )
    _require(
        np.allclose(
            metrics["mean_squared_error_area_weighted"] * metrics["evaluation_area_mm2"],
            metrics["integrated_squared_error_mm2"],
            rtol=2e-12,
            atol=2e-11,
        ),
        "integrated and mean squared errors disagree",
    )
    for column in (
        "rmse_area_weighted",
        "mae_area_weighted",
        "binned_score_calibration_error",
        "uncensored_state_out_of_range_fraction",
        "dice_voltage_le_0p1",
        "balanced_accuracy_voltage_le_0p1",
    ):
        _require(((metrics[column] >= 0.0) & (metrics[column] <= 1.0)).all(), f"{column} leaves [0,1]")
    _require((metrics["solver_relative_residual"] <= 2e-4).all(), "solver residual too large")
    _require((phase_rows["maximum_mass_source_defect"] <= 2e-4).all(), "mass defect too large")
    _require((phase_rows["maximum_graph_projection_residual"] <= 2e-4).all(), "graph projection residual too large")
    _require(
        (metrics.loc[metrics["method"] == "screened", "pseudo_horizon"] == 0.0).all(),
        "screened rows must have zero phase horizon",
    )
    _require(
        np.allclose(
            metrics.loc[metrics["method"] != "screened", "pseudo_horizon"],
            design.PHASE_HORIZON,
        ),
        "phase rows do not use the locked common horizon",
    )

    expected_capacity_keys = {
        (patient, mask, method, pv)
        for patient in design.PATIENT_IDS
        for mask, pvs in design.MASK_GROUPS.items()
        for method in methods
        for pv in pvs
    }
    _require(len(capacity) == 72, "capacity table must have 72 rows")
    _require(
        _key_set(capacity, ("patient_id", "mask", "method", "pv_label"))
        == expected_capacity_keys,
        "capacity key grid is incomplete",
    )
    _finite(
        capacity,
        [
            "normalised_capacity",
            "reference_normalised_capacity",
            "capacity_absolute_error",
            "capacity_absolute_log_ratio",
            "capacity_relative_residual",
            "reference_capacity_relative_residual",
            "widest_viable_arc_mm",
            "reference_widest_viable_arc_mm",
        ],
    )
    for column in ("normalised_capacity", "reference_normalised_capacity"):
        _require(((capacity[column] >= 1e-3 - 1e-8) & (capacity[column] <= 1.0 + 1e-8)).all(), f"{column} violates coefficient bounds")
    _require((capacity["capacity_relative_residual"] <= 1e-9).all(), "capacity residual too large")
    _require((capacity["reference_capacity_relative_residual"] <= 1e-9).all(), "reference capacity residual too large")

    expected_summary_keys = {
        (patient, method) for patient in design.PATIENT_IDS for method in methods
    }
    _require(len(summary) == 18, "patient summary must have 18 rows")
    _require(_key_set(summary, ("patient_id", "method")) == expected_summary_keys, "summary keys incomplete")
    recomputed = design._patient_summary(metrics, capacity)
    merged = summary.merge(recomputed, on=["patient_id", "method"], suffixes=("_file", "_new"), validate="one_to_one")
    for column in recomputed.columns.difference(["patient_id", "method"]):
        _require(
            np.allclose(merged[f"{column}_file"], merged[f"{column}_new"], equal_nan=True, rtol=2e-12, atol=2e-12),
            f"patient summary column {column} is not reproducible",
        )
    _require(len(contrasts) == 16, "contrast table must have 16 rows")
    _require((contrasts["n_patients"] == 6).all(), "a contrast is not patient-level n=6")
    _require(((contrasts["exact_two_sided_sign_flip_p"] >= 0.0) & (contrasts["exact_two_sided_sign_flip_p"] <= 1.0)).all(), "invalid exact p value")

    with np.load(paths["representative"]) as arrays:
        required = {
            "points", "triangles", "bi_mv", "reference_score", "signed_reference",
            "evaluation_mask", "guard_mask", "blackout_mask", "contact_nodes",
            "confidence", "forcing", "screened_state", "passive_state", "graph_state",
            *(f"boundary_{label}" for label in design.BOUNDARY_TAGS),
        }
        _require(required.issubset(arrays.files), "representative NPZ lacks a required array")
        n = len(arrays["points"])
        _require(arrays["triangles"].shape[1] == 3, "representative cells are not triangles")
        for key in ("bi_mv", "reference_score", "signed_reference", "confidence", "forcing", "screened_state", "passive_state", "graph_state"):
            _require(arrays[key].shape == (n,), f"representative {key} has wrong shape")
        blackout = arrays["blackout_mask"].astype(bool)
        _require(np.max(arrays["confidence"][blackout]) == 0.0, "NPZ blackout confidence is nonzero")
        _require(np.max(np.abs(arrays["forcing"][blackout])) == 0.0, "NPZ blackout forcing is nonzero")

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    _require(provenance["dataset"]["doi"] == design.DATASET_DOI, "wrong dataset DOI")
    _require(provenance["source_paper"]["doi"] == design.SOURCE_PAPER_DOI, "wrong paper DOI")
    _require(provenance["dataset"]["license"] == design.LICENSE, "wrong dataset license")
    _require(provenance["numerics"]["surface_diffusion_tensor"] == "isotropic scalar a=1", "tensor provenance missing")
    _require(provenance["numerics"]["patient_fibres_used"] is False, "fibre use is misreported")
    _require("not a probability" in provenance["target"]["interpretation"], "score interpretation missing")
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    actual_frames = {
        "zenodo_pvi_mesh_inventory.csv": inventory,
        "zenodo_pvi_boundary_transfer.csv": boundary,
        "zenodo_pvi_reconstruction_metrics.csv": metrics,
        "zenodo_pvi_capacity.csv": capacity,
        "zenodo_pvi_patient_summary.csv": summary,
        "zenodo_pvi_patient_contrasts.csv": contrasts,
    }
    for filename, frame in actual_frames.items():
        _require(schema["tables"][filename]["columns"] == list(frame.columns), f"schema columns disagree for {filename}")
    _require("integrated_squared_error_mm2" in schema["endpoint_definitions"], "ISE definition missing")
    _require("uncensored_state_out_of_range_fraction" in schema["endpoint_definitions"], "uncensored-state definition missing")
    print("all Zenodo patient-surface checks passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate(args.data_dir, args.figure_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
