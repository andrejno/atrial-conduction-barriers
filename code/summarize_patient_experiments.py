#!/usr/bin/env python3
"""Locked, outcome-free summaries for the 82-case patient experiments P1 and P2.

The script accepts only the immutable all-cohort outputs written by
``run_patient_reconstruction.py``.  It checks the analysis lock, the selected
numerical configuration, every input digest, the outcome-blinding flag, and
the complete patient/PV/rotation/method Cartesian products before calculating
anything.  It has no clinical-outcome input and uses only the Python standard
library.

The P1 primary endpoint pools the guarded-region Brier numerator and area over
four PVs and twelve rotations within each patient.  The primary comparison is
the graph reconstruction minus the passive ablation from the same locked
configuration.  P2 uses the locked patient-level capacity-fidelity error,

    median_rotation mean_PV abs(log(C_reconstructed / C_full)).

Existing output files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0"
DEVELOPMENT_IDS = tuple(f"ID{value:03d}" for value in range(21, 88))
LOCKED_HOLDOUT_IDS = tuple(f"ID{value:03d}" for value in range(1, 16))
EXPECTED_IDS = LOCKED_HOLDOUT_IDS + DEVELOPMENT_IDS
PV_LABELS = ("LSPV", "LIPV", "RSPV", "RIPV")
ESTIMATORS = (
    "prevalence",
    "nearest_neighbour",
    "screened",
    "passive",
    "graph",
)
PRIMARY_CONDUCTIVITY_MAP = "eta(s)=1e-3+(1-1e-3)/(1+exp(8s))"
FULL_REFERENCE_SCORE = "signed_q_2b_minus_1"
RECONSTRUCTION_SCORE = "unclipped_u"
BOOTSTRAP_SEED = 20250903
BOOTSTRAP_RESAMPLES = 100_000
TIE_TOLERANCE = 1.0e-12


RECONSTRUCTION_METRIC_COLUMNS = (
    "patient_id",
    "partition",
    "pv_label",
    "replicate",
    "mask_variant",
    "sector_fraction",
    "band_width_mm",
    "evaluation_area_mm2",
    "configuration_id",
    "estimator",
    "rmse_area_weighted",
    "mae_area_weighted",
    "brier_area_weighted",
    "dice_barrier",
    "balanced_accuracy",
    "boundary_distance_mm",
    "topology_match",
    "normalised_capacity",
    "full_normalised_capacity",
    "capacity_relative_error",
    "absolute_log_capacity_ratio",
    "solver_converged",
    "solver_residual",
)

PHENOTYPE_COLUMNS = (
    "patient_id",
    "partition",
    "configuration_id",
    "estimator",
    "replicate",
    "mask_variant",
    "pv_label",
    "annulus_width_mm",
    "conductivity_map",
    "conductivity_score_source",
    "normalised_capacity",
    "capacity_relative_residual",
    "widest_viable_arc_mm",
    "viable_arc_count",
    "minimum_barrier_score",
    "phenotype_complete",
)

P1_COLUMNS = (
    "patient_id",
    "partition",
    "n_pv_rotation_units",
    "total_evaluation_area_mm2",
    "prevalence_brier_area_weighted",
    "nearest_neighbour_brier_area_weighted",
    "screened_brier_area_weighted",
    "passive_brier_area_weighted",
    "graph_brier_area_weighted",
    "graph_minus_passive_brier",
    "graph_vs_passive",
)

P2_COLUMNS = (
    "patient_id",
    "partition",
    "n_pv_rotation_units",
    "prevalence_capacity_fidelity_error",
    "nearest_neighbour_capacity_fidelity_error",
    "screened_capacity_fidelity_error",
    "passive_capacity_fidelity_error",
    "graph_capacity_fidelity_error",
    "graph_minus_passive_capacity_fidelity_error",
    "graph_vs_passive",
    "full_barrier_strength",
    "passive_barrier_strength",
    "graph_barrier_strength",
    "passive_minus_full_barrier_strength",
    "graph_minus_full_barrier_strength",
    "full_most_permissive_pv",
    "passive_most_permissive_pv",
    "graph_most_permissive_pv",
    "passive_most_permissive_pv_match",
    "graph_most_permissive_pv_match",
    "passive_gap_count_mae",
    "graph_gap_count_mae",
    "passive_widest_viable_arc_mae_mm",
    "graph_widest_viable_arc_mae_mm",
)


class PatientSummaryError(RuntimeError):
    """Raised when a locked input or result contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PatientSummaryError(f"cannot hash required input {path}: {exc}") from exc
    return digest.hexdigest()


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PatientSummaryError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise PatientSummaryError(f"{description} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_json_no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise PatientSummaryError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PatientSummaryError(f"{description} root must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise PatientSummaryError(
            f"{context} has an unexpected key set; missing={missing}, extra={extra}"
        )


def _mapping(value: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    child = value.get(key)
    if not isinstance(child, dict):
        raise PatientSummaryError(f"{context}.{key} must be an object")
    return child


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive(value: Any, context: str) -> float:
    if not _is_number(value) or float(value) <= 0.0:
        raise PatientSummaryError(f"{context} must be finite and positive")
    return float(value)


def _sha_text(value: Any, context: str, lengths: Sequence[int] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or re.fullmatch(r"[0-9a-f]+", value) is None
    ):
        allowed = " or ".join(str(length) for length in lengths)
        raise PatientSummaryError(f"{context} must be a lowercase {allowed}-digit digest")
    return value


def _completed_text(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "REPLACE" in value.upper()
    ):
        raise PatientSummaryError(f"{context} is not completed")
    return value


def _partition(patient_id: str) -> str:
    if patient_id in DEVELOPMENT_IDS:
        return "development"
    if patient_id in LOCKED_HOLDOUT_IDS:
        return "locked_holdout"
    raise PatientSummaryError(f"unexpected patient identifier {patient_id!r}")


def _close(left: float, right: float, *, rtol: float = 5.0e-12, atol: float = 5.0e-13) -> bool:
    return abs(left - right) <= atol + rtol * max(abs(left), abs(right))


def _validate_configuration(configuration: Mapping[str, Any], context: str) -> None:
    _exact_keys(
        configuration,
        {
            "configuration_id",
            "reference_length_mm",
            "coordinate_rule",
            "observation_confidence",
            "probability_threshold",
            "terminal_readout",
            "terminal_average_steps",
            "screened",
            "phase",
            "capacity",
            "conductivity",
        },
        context,
    )
    _completed_text(configuration.get("configuration_id"), f"{context}.configuration_id")
    if configuration.get("reference_length_mm") != 1.0:
        raise PatientSummaryError(f"{context}.reference_length_mm must equal 1.0")
    if configuration.get("coordinate_rule") != "x_hat=x/reference_length_mm":
        raise PatientSummaryError(f"{context}.coordinate_rule differs from the locked rule")
    _positive(configuration.get("observation_confidence"), f"{context}.observation_confidence")
    threshold = _positive(configuration.get("probability_threshold"), f"{context}.probability_threshold")
    if threshold >= 1.0:
        raise PatientSummaryError(f"{context}.probability_threshold must be below one")
    if configuration.get("terminal_readout") != "mean_last_k_post_step_states":
        raise PatientSummaryError(f"{context}.terminal_readout differs from the locked rule")
    terminal_steps = configuration.get("terminal_average_steps")
    if not isinstance(terminal_steps, int) or isinstance(terminal_steps, bool) or terminal_steps < 1:
        raise PatientSummaryError(f"{context}.terminal_average_steps must be a positive integer")

    screened = _mapping(configuration, "screened", context)
    _exact_keys(
        screened,
        {
            "epsilon",
            "length_scale_dimensionless",
            "linear_method",
            "linear_tolerance",
            "linear_absolute_tolerance",
            "linear_max_iterations",
        },
        f"{context}.screened",
    )
    for name in ("epsilon", "length_scale_dimensionless", "linear_tolerance", "linear_absolute_tolerance"):
        _positive(screened.get(name), f"{context}.screened.{name}")
    if screened.get("linear_method") not in {"direct", "cg"}:
        raise PatientSummaryError(f"{context}.screened.linear_method is invalid")
    if not isinstance(screened.get("linear_max_iterations"), int) or isinstance(screened.get("linear_max_iterations"), bool) or screened["linear_max_iterations"] < 1:
        raise PatientSummaryError(f"{context}.screened.linear_max_iterations must be positive")

    phase = _mapping(configuration, "phase", context)
    _exact_keys(
        phase,
        {
            "mu",
            "graph_nu",
            "dt",
            "nsteps",
            "rho_factor",
            "admm_tolerance",
            "admm_max_iterations",
            "linear_method",
            "linear_tolerance",
            "linear_absolute_tolerance",
            "linear_max_iterations",
        },
        f"{context}.phase",
    )
    for name in (
        "mu",
        "graph_nu",
        "dt",
        "rho_factor",
        "admm_tolerance",
        "linear_tolerance",
        "linear_absolute_tolerance",
    ):
        _positive(phase.get(name), f"{context}.phase.{name}")
    for name in ("nsteps", "admm_max_iterations", "linear_max_iterations"):
        value = phase.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise PatientSummaryError(f"{context}.phase.{name} must be a positive integer")
    if terminal_steps > phase["nsteps"]:
        raise PatientSummaryError(f"{context}.terminal_average_steps exceeds nsteps")
    if phase.get("linear_method") not in {"direct", "cg"}:
        raise PatientSummaryError(f"{context}.phase.linear_method is invalid")

    capacity = _mapping(configuration, "capacity", context)
    _exact_keys(
        capacity,
        {"annulus_width_mm", "outer_electrode_rule", "relative_residual_max"},
        f"{context}.capacity",
    )
    if capacity.get("annulus_width_mm") != 10.0:
        raise PatientSummaryError(f"{context}.capacity.annulus_width_mm must equal 10.0")
    if capacity.get("outer_electrode_rule") != "inner_vertex_frontier_of_geodesic_sublevel":
        raise PatientSummaryError(f"{context}.capacity.outer_electrode_rule is invalid")
    if _positive(capacity.get("relative_residual_max"), f"{context}.capacity.relative_residual_max") > 1.0e-10:
        raise PatientSummaryError(f"{context}.capacity.relative_residual_max exceeds 1e-10")

    conductivity = _mapping(configuration, "conductivity", context)
    _exact_keys(
        conductivity,
        {"formula", "full_reference_score", "reconstruction_score"},
        f"{context}.conductivity",
    )
    expected = {
        "formula": PRIMARY_CONDUCTIVITY_MAP,
        "full_reference_score": FULL_REFERENCE_SCORE,
        "reconstruction_score": RECONSTRUCTION_SCORE,
    }
    for name, wanted in expected.items():
        if conductivity.get(name) != wanted:
            raise PatientSummaryError(f"{context}.conductivity.{name} differs from the locked rule")


def _load_locked_numerics(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    numerics = _read_json(path, "patient numerics")
    _exact_keys(numerics, {"schema_version", "development_candidates", "selection"}, "patient_numerics")
    if numerics.get("schema_version") != SCHEMA_VERSION:
        raise PatientSummaryError("patient_numerics.schema_version must equal '1.0'")
    candidates = numerics.get("development_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PatientSummaryError("patient_numerics.development_candidates must be non-empty")
    by_id: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise PatientSummaryError(f"development_candidates[{index}] must be an object")
        _validate_configuration(candidate, f"development_candidates[{index}]")
        identifier = candidate["configuration_id"]
        if identifier in by_id:
            raise PatientSummaryError("development configuration identifiers are not unique")
        by_id[identifier] = candidate
    selection = _mapping(numerics, "selection", "patient_numerics")
    _exact_keys(
        selection,
        {
            "status",
            "rule",
            "selected_without_holdout",
            "complexity_order",
            "selected_configuration",
            "selection_evidence",
        },
        "patient_numerics.selection",
    )
    if selection.get("status") != "LOCKED":
        raise PatientSummaryError("patient numerical selection is not LOCKED")
    if selection.get("rule") != "one_standard_error_patient_level_brier":
        raise PatientSummaryError("patient numerical selection rule differs from the lock")
    if selection.get("selected_without_holdout") is not True:
        raise PatientSummaryError("numerical selection does not certify holdout blinding")
    order = selection.get("complexity_order")
    if (
        not isinstance(order, list)
        or any(not isinstance(item, str) for item in order)
        or len(order) != len(set(order))
        or set(order) != set(by_id)
    ):
        raise PatientSummaryError("selection.complexity_order must list every candidate once")
    selected = selection.get("selected_configuration")
    if not isinstance(selected, dict):
        raise PatientSummaryError("LOCKED numerical selection has no selected configuration")
    _validate_configuration(selected, "patient_numerics.selection.selected_configuration")
    identifier = selected["configuration_id"]
    if identifier not in by_id or selected != by_id[identifier]:
        raise PatientSummaryError("selected configuration does not exactly match its candidate")
    evidence = selection.get("selection_evidence")
    if not isinstance(evidence, dict):
        raise PatientSummaryError("LOCKED numerical selection has no selection_evidence")
    _exact_keys(
        evidence,
        {"source_draft_numerics_sha256", "selection_report_sha256"},
        "patient_numerics.selection.selection_evidence",
    )
    for name in ("source_draft_numerics_sha256", "selection_report_sha256"):
        digest = _sha_text(
            evidence.get(name), f"patient_numerics.selection.selection_evidence.{name}"
        )
        if digest == "0" * 64:
            raise PatientSummaryError(
                f"patient_numerics.selection.selection_evidence.{name} cannot be an empty placeholder digest"
            )
    return numerics, selected


def _load_analysis_lock(path: Path, expected_sha256: str, numerics_path: Path) -> dict[str, Any]:
    _sha_text(expected_sha256, "--expected-lock-sha256")
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise PatientSummaryError("analysis-lock checksum does not match")
    lock = _read_json(path, "analysis lock")
    _exact_keys(
        lock,
        {
            "schema_version",
            "status",
            "protocol",
            "source",
            "outcome_blinding",
            "primary_settings",
            "frozen_phenotype",
            "numerical_acceptance",
            "locked_outputs",
        },
        "analysis_lock",
    )
    if lock.get("schema_version") != SCHEMA_VERSION or lock.get("status") != "LOCKED":
        raise PatientSummaryError("analysis lock must be schema 1.0 with status LOCKED")

    protocol = _mapping(lock, "protocol", "analysis_lock")
    _exact_keys(protocol, {"path", "sha256"}, "analysis_lock.protocol")
    _completed_text(protocol.get("path"), "analysis_lock.protocol.path")
    _sha_text(protocol.get("sha256"), "analysis_lock.protocol.sha256")

    source = _mapping(lock, "source", "analysis_lock")
    _exact_keys(
        source,
        {
            "git_commit_or_tree_hash",
            "environment_lock_sha256",
            "patient_input_manifest_sha256",
            "patient_numerics_manifest_sha256",
        },
        "analysis_lock.source",
    )
    _sha_text(source.get("git_commit_or_tree_hash"), "analysis_lock.source.git_commit_or_tree_hash", (40, 64))
    for name in ("environment_lock_sha256", "patient_input_manifest_sha256", "patient_numerics_manifest_sha256"):
        _sha_text(source.get(name), f"analysis_lock.source.{name}")
    if source["patient_numerics_manifest_sha256"] != _sha256(numerics_path):
        raise PatientSummaryError("analysis lock does not identify the supplied patient numerics")

    blinding = _mapping(lock, "outcome_blinding", "analysis_lock")
    _exact_keys(
        blinding,
        {"clinical_manifest_unavailable_during_lock", "confirmed_by", "confirmed_utc"},
        "analysis_lock.outcome_blinding",
    )
    if blinding.get("clinical_manifest_unavailable_during_lock") is not True:
        raise PatientSummaryError("analysis lock does not certify outcome blinding")
    _completed_text(blinding.get("confirmed_by"), "analysis_lock.outcome_blinding.confirmed_by")
    timestamp = _completed_text(blinding.get("confirmed_utc"), "analysis_lock.outcome_blinding.confirmed_utc")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z", timestamp) is None:
        raise PatientSummaryError("analysis_lock.outcome_blinding.confirmed_utc is not UTC ISO-8601")

    primary = _mapping(lock, "primary_settings", "analysis_lock")
    expected_primary = {
        "development_size": 67,
        "locked_holdout_size": 15,
        "mask_seed": 20250903,
        "mask_replicates": 12,
        "masked_boundary_arc_fraction": 0.5,
        "annulus_width_mm": 10.0,
        "guard_width_mm": 1.5,
        "reconstruction_endpoint": "patient_area_weighted_brier",
        "phenotype_fidelity_endpoint": "patient_median_mean_abs_log_capacity_ratio",
        "recurrence_endpoint": "locked_holdout_delta_log_loss_m1_minus_m0",
    }
    _exact_keys(primary, set(expected_primary), "analysis_lock.primary_settings")
    for name, wanted in expected_primary.items():
        if primary.get(name) != wanted:
            raise PatientSummaryError(f"analysis_lock.primary_settings.{name} differs from the frozen value")

    phenotype = _mapping(lock, "frozen_phenotype", "analysis_lock")
    expected_phenotype = {
        "name": "B_i",
        "estimator": "direct_graph_reconstruction_from_primary_50_percent_masks",
        "capacity_per_rotation": "normalized_capacity",
        "within_pv_rotation_aggregation": "median",
        "across_pv_aggregation": "maximum",
        "operation_order": [
            "normalized_capacity_per_rotation_per_pv",
            "median_over_12_rotations_within_each_pv",
            "maximum_over_pvs",
            "negative_log",
        ],
        "definition": "-log(max_pv(median_12_rotations(normalized_capacity)))",
        "conductivity_map": PRIMARY_CONDUCTIVITY_MAP,
        "full_reference_score": FULL_REFERENCE_SCORE,
        "reconstruction_score": RECONSTRUCTION_SCORE,
        "probability_clipping_only": True,
        "log_offset": 0.0,
        "strict_positive_capacity_required": True,
        "higher_means": "stronger_modeled_barrier",
    }
    _exact_keys(phenotype, set(expected_phenotype), "analysis_lock.frozen_phenotype")
    for name, wanted in expected_phenotype.items():
        if phenotype.get(name) != wanted:
            raise PatientSummaryError(f"analysis_lock.frozen_phenotype.{name} differs from the frozen value")

    acceptance = _mapping(lock, "numerical_acceptance", "analysis_lock")
    expected_acceptance = {
        "capacity_relative_residual_max": 1.0e-10,
        "capacity_two_finest_relative_change_max": 0.05,
        "whole_surface_valid_area_fraction_min": 0.90,
        "pv_annulus_valid_area_fraction_min": 0.95,
    }
    _exact_keys(acceptance, set(expected_acceptance), "analysis_lock.numerical_acceptance")
    for name, wanted in expected_acceptance.items():
        if acceptance.get(name) != wanted:
            raise PatientSummaryError(f"analysis_lock.numerical_acceptance.{name} differs from the frozen value")

    outputs = _mapping(lock, "locked_outputs", "analysis_lock")
    _exact_keys(outputs, {"directory", "created_utc", "manifest_sha256"}, "analysis_lock.locked_outputs")
    _completed_text(outputs.get("directory"), "analysis_lock.locked_outputs.directory")
    output_timestamp = _completed_text(outputs.get("created_utc"), "analysis_lock.locked_outputs.created_utc")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z", output_timestamp) is None:
        raise PatientSummaryError("analysis_lock.locked_outputs.created_utc is not UTC ISO-8601")
    _sha_text(outputs.get("manifest_sha256"), "analysis_lock.locked_outputs.manifest_sha256")
    return lock


def _load_run_provenance(
    path: Path,
    *,
    metrics_path: Path,
    phenotype_path: Path,
    mask_manifest_path: Path,
    numerics_path: Path,
    lock: Mapping[str, Any],
    lock_sha256: str,
    configuration_id: str,
) -> dict[str, Any]:
    provenance = _read_json(path, "P1/P2 run provenance")
    expected_keys = {
        "schema_version",
        "partition",
        "n_patients",
        "configuration_id",
        "configuration_mode",
        "patient_input_manifest_sha256",
        "patient_numerics_manifest_sha256",
        "analysis_lock_sha256",
        "mask_manifest_sha256",
        "reconstruction_metrics_sha256",
        "phenotype_table_sha256",
        "outcomes_accessed",
    }
    _exact_keys(provenance, expected_keys, "run_provenance")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "partition": "all",
        "configuration_id": configuration_id,
        "configuration_mode": "locked_selection",
        "patient_input_manifest_sha256": lock["source"]["patient_input_manifest_sha256"],
        "patient_numerics_manifest_sha256": _sha256(numerics_path),
        "analysis_lock_sha256": lock_sha256,
        "mask_manifest_sha256": _sha256(mask_manifest_path),
        "reconstruction_metrics_sha256": _sha256(metrics_path),
        "phenotype_table_sha256": _sha256(phenotype_path),
    }
    for name, wanted in exact.items():
        if provenance.get(name) != wanted:
            raise PatientSummaryError(f"run_provenance.{name} does not match the locked all-82 input")
    if (
        not isinstance(provenance.get("n_patients"), int)
        or isinstance(provenance.get("n_patients"), bool)
        or provenance["n_patients"] != 82
    ):
        raise PatientSummaryError("run_provenance.n_patients must be the integer 82")
    if provenance.get("outcomes_accessed") is not False:
        raise PatientSummaryError("run_provenance.outcomes_accessed must be the boolean false")
    return provenance


def _read_csv(path: Path, columns: Sequence[str], description: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise PatientSummaryError(f"{description} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or tuple(reader.fieldnames) != tuple(columns):
                raise PatientSummaryError(f"{description} header differs from the immutable schema")
            rows = list(reader)
            expected = set(columns)
            for number, row in enumerate(rows, start=2):
                if set(row) != expected or any(value is None for value in row.values()):
                    raise PatientSummaryError(
                        f"{description} row {number} has missing or trailing cells"
                    )
            return rows
    except OSError as exc:
        raise PatientSummaryError(f"cannot read {description} {path}: {exc}") from exc


def _float(row: Mapping[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise PatientSummaryError(f"{context}.{field} is not numeric") from exc
    if not math.isfinite(value):
        raise PatientSummaryError(f"{context}.{field} is not finite")
    return value


def _int(row: Mapping[str, str], field: str, context: str) -> int:
    text = row.get(field, "")
    if re.fullmatch(r"-?[0-9]+", text) is None:
        raise PatientSummaryError(f"{context}.{field} is not an integer")
    return int(text)


def _bounded(value: float, lower: float, upper: float, context: str) -> None:
    if value < lower or value > upper:
        raise PatientSummaryError(f"{context} is outside [{lower:g},{upper:g}]")


def _validate_metric_rows(
    rows: Sequence[dict[str, str]],
    configuration: Mapping[str, Any],
) -> dict[tuple[str, str, int, str], dict[str, float]]:
    configuration_id = str(configuration["configuration_id"])
    annulus_width = float(configuration["capacity"]["annulus_width_mm"])
    screened_residual_limit = max(
        100.0 * float(configuration["screened"]["linear_tolerance"]), 1.0e-10
    )
    phase_residual_limit = max(
        20.0 * float(configuration["phase"]["admm_tolerance"]),
        100.0 * float(configuration["phase"]["linear_tolerance"]),
        1.0e-9,
    )
    expected_count = len(EXPECTED_IDS) * len(PV_LABELS) * 12 * len(ESTIMATORS)
    if len(rows) != expected_count:
        raise PatientSummaryError(
            f"reconstruction metrics contain {len(rows)} rows; expected exactly {expected_count}"
        )
    parsed: dict[tuple[str, str, int, str], dict[str, float]] = {}
    per_mask_area: dict[tuple[str, str, int], float] = {}
    per_mask_full_capacity: dict[tuple[str, str, int], float] = {}
    for number, row in enumerate(rows, start=2):
        context = f"reconstruction_metrics row {number}"
        patient = row["patient_id"]
        partition = _partition(patient)
        if row["partition"] != partition:
            raise PatientSummaryError(f"{context}.partition is incorrect")
        pv = row["pv_label"]
        if pv not in PV_LABELS:
            raise PatientSummaryError(f"{context}.pv_label is not canonical")
        replicate = _int(row, "replicate", context)
        if not 0 <= replicate < 12:
            raise PatientSummaryError(f"{context}.replicate is outside 0--11")
        estimator = row["estimator"]
        if estimator not in ESTIMATORS:
            raise PatientSummaryError(f"{context}.estimator is not part of the locked comparison")
        if row["mask_variant"] != "primary":
            raise PatientSummaryError(f"{context}.mask_variant must equal primary")
        if row["configuration_id"] != configuration_id:
            raise PatientSummaryError(f"{context}.configuration_id is not selected")
        sector = _float(row, "sector_fraction", context)
        width = _float(row, "band_width_mm", context)
        if not _close(sector, 0.5) or not _close(width, annulus_width):
            raise PatientSummaryError(f"{context} differs from the locked 50%/10-mm mask")
        area = _float(row, "evaluation_area_mm2", context)
        if area <= 0.0:
            raise PatientSummaryError(f"{context}.evaluation_area_mm2 is not positive")
        brier = _float(row, "brier_area_weighted", context)
        rmse = _float(row, "rmse_area_weighted", context)
        mae = _float(row, "mae_area_weighted", context)
        dice = _float(row, "dice_barrier", context)
        for name, value in (("brier", brier), ("rmse", rmse), ("mae", mae), ("dice", dice)):
            _bounded(value, 0.0, 1.0 + 1.0e-12, f"{context}.{name}")
        if not _close(rmse * rmse, brier, rtol=2.0e-10, atol=2.0e-12):
            raise PatientSummaryError(f"{context} has inconsistent RMSE and Brier values")
        balanced = row["balanced_accuracy"]
        if balanced != "":
            _bounded(_float(row, "balanced_accuracy", context), 0.0, 1.0, f"{context}.balanced_accuracy")
        boundary = row["boundary_distance_mm"]
        if boundary != "" and _float(row, "boundary_distance_mm", context) < 0.0:
            raise PatientSummaryError(f"{context}.boundary_distance_mm is negative")
        topology = _int(row, "topology_match", context)
        if topology not in {0, 1}:
            raise PatientSummaryError(f"{context}.topology_match must be zero or one")
        capacity = _float(row, "normalised_capacity", context)
        full_capacity = _float(row, "full_normalised_capacity", context)
        for name, value in (("normalised_capacity", capacity), ("full_normalised_capacity", full_capacity)):
            if value < 1.0e-3 - 5.0e-10 or value > 1.0 + 5.0e-10:
                raise PatientSummaryError(f"{context}.{name} violates conductivity bounds")
        relative_error = _float(row, "capacity_relative_error", context)
        log_error = _float(row, "absolute_log_capacity_ratio", context)
        if not _close(relative_error, abs(capacity - full_capacity) / full_capacity):
            raise PatientSummaryError(f"{context}.capacity_relative_error is inconsistent")
        if not _close(log_error, abs(math.log(capacity / full_capacity))):
            raise PatientSummaryError(f"{context}.absolute_log_capacity_ratio is inconsistent")
        if _int(row, "solver_converged", context) != 1:
            raise PatientSummaryError(f"{context}.solver_converged is not one")
        residual = _float(row, "solver_residual", context)
        if residual < 0.0:
            raise PatientSummaryError(f"{context}.solver_residual is negative")
        if estimator in {"prevalence", "nearest_neighbour"} and residual != 0.0:
            raise PatientSummaryError(
                f"{context}.solver_residual must be zero for an algebraic comparator"
            )
        if estimator == "screened" and residual > screened_residual_limit:
            raise PatientSummaryError(
                f"{context}.solver_residual exceeds the selected screened tolerance"
            )
        if estimator in {"passive", "graph"} and residual > phase_residual_limit:
            raise PatientSummaryError(
                f"{context}.solver_residual exceeds the selected phase tolerance"
            )
        key = (patient, pv, replicate, estimator)
        if key in parsed:
            raise PatientSummaryError(f"duplicate reconstruction metric for {key}")
        parsed[key] = {
            "area": area,
            "brier": brier,
            "mae": mae,
            "rmse": rmse,
            "dice": dice,
            "capacity": capacity,
            "full_capacity": full_capacity,
            "log_capacity_error": log_error,
        }
        mask_key = (patient, pv, replicate)
        if mask_key in per_mask_area and not _close(per_mask_area[mask_key], area):
            raise PatientSummaryError(f"methods do not share evaluation area for {mask_key}")
        if mask_key in per_mask_full_capacity and not _close(per_mask_full_capacity[mask_key], full_capacity):
            raise PatientSummaryError(f"methods do not share full capacity for {mask_key}")
        per_mask_area[mask_key] = area
        per_mask_full_capacity[mask_key] = full_capacity
    expected_keys = {
        (patient, pv, replicate, estimator)
        for patient in EXPECTED_IDS
        for pv in PV_LABELS
        for replicate in range(12)
        for estimator in ESTIMATORS
    }
    if set(parsed) != expected_keys:
        missing = sorted(expected_keys - set(parsed))[:5]
        extra = sorted(set(parsed) - expected_keys)[:5]
        raise PatientSummaryError(f"metric Cartesian product is incomplete; missing={missing}, extra={extra}")
    return parsed


def _validate_phenotype_rows(
    rows: Sequence[dict[str, str]],
    configuration_id: str,
    annulus_width: float,
    residual_limit: float,
    metrics: Mapping[tuple[str, str, int, str], Mapping[str, float]],
) -> tuple[
    dict[tuple[str, str], dict[str, float | int]],
    dict[tuple[str, str, int, str], dict[str, float | int]],
]:
    expected_count = len(EXPECTED_IDS) * len(PV_LABELS) * (1 + 12 * len(ESTIMATORS))
    if len(rows) != expected_count:
        raise PatientSummaryError(
            f"long-form phenotype contains {len(rows)} rows; expected exactly {expected_count}"
        )
    full: dict[tuple[str, str], dict[str, float | int]] = {}
    primary: dict[tuple[str, str, int, str], dict[str, float | int]] = {}
    for number, row in enumerate(rows, start=2):
        context = f"phenotype row {number}"
        patient = row["patient_id"]
        partition = _partition(patient)
        if row["partition"] != partition:
            raise PatientSummaryError(f"{context}.partition is incorrect")
        if row["configuration_id"] != configuration_id:
            raise PatientSummaryError(f"{context}.configuration_id is not selected")
        pv = row["pv_label"]
        if pv not in PV_LABELS:
            raise PatientSummaryError(f"{context}.pv_label is not canonical")
        if row["conductivity_map"] != PRIMARY_CONDUCTIVITY_MAP:
            raise PatientSummaryError(f"{context}.conductivity_map differs from the lock")
        if not _close(_float(row, "annulus_width_mm", context), annulus_width):
            raise PatientSummaryError(f"{context}.annulus_width_mm differs from the lock")
        capacity = _float(row, "normalised_capacity", context)
        if capacity < 1.0e-3 - 5.0e-10 or capacity > 1.0 + 5.0e-10:
            raise PatientSummaryError(f"{context}.normalised_capacity violates conductivity bounds")
        residual = _float(row, "capacity_relative_residual", context)
        if residual < 0.0 or residual > residual_limit:
            raise PatientSummaryError(f"{context}.capacity_relative_residual exceeds the lock")
        widest = _float(row, "widest_viable_arc_mm", context)
        if widest < 0.0:
            raise PatientSummaryError(f"{context}.widest_viable_arc_mm is negative")
        gap_count = _int(row, "viable_arc_count", context)
        if gap_count < 0:
            raise PatientSummaryError(f"{context}.viable_arc_count is negative")
        minimum = _float(row, "minimum_barrier_score", context)
        if _int(row, "phenotype_complete", context) != 1:
            raise PatientSummaryError(f"{context}.phenotype_complete is not one")
        estimator = row["estimator"]
        values: dict[str, float | int] = {
            "capacity": capacity,
            "widest": widest,
            "gap_count": gap_count,
            "minimum": minimum,
        }
        if estimator == "full_imaging_reference":
            if (
                _int(row, "replicate", context) != -1
                or row["mask_variant"] != "full"
                or row["conductivity_score_source"] != FULL_REFERENCE_SCORE
            ):
                raise PatientSummaryError(f"{context} is not a valid full-reference row")
            key = (patient, pv)
            if key in full:
                raise PatientSummaryError(f"duplicate full phenotype row for {key}")
            full[key] = values
        else:
            if estimator not in ESTIMATORS:
                raise PatientSummaryError(f"{context}.estimator is not part of the locked comparison")
            replicate = _int(row, "replicate", context)
            if not 0 <= replicate < 12:
                raise PatientSummaryError(f"{context}.replicate is outside 0--11")
            if row["mask_variant"] != "primary" or row["conductivity_score_source"] != RECONSTRUCTION_SCORE:
                raise PatientSummaryError(f"{context} is not a primary reconstruction row")
            key = (patient, pv, replicate, estimator)
            if key in primary:
                raise PatientSummaryError(f"duplicate primary phenotype row for {key}")
            primary[key] = values

    expected_full = {(patient, pv) for patient in EXPECTED_IDS for pv in PV_LABELS}
    expected_primary = {
        (patient, pv, replicate, estimator)
        for patient in EXPECTED_IDS
        for pv in PV_LABELS
        for replicate in range(12)
        for estimator in ESTIMATORS
    }
    if set(full) != expected_full:
        raise PatientSummaryError("full-reference phenotype is not exactly 82 patients by 4 PVs")
    if set(primary) != expected_primary:
        missing = sorted(expected_primary - set(primary))[:5]
        extra = sorted(set(primary) - expected_primary)[:5]
        raise PatientSummaryError(f"primary phenotype Cartesian product is incomplete; missing={missing}, extra={extra}")

    for key, metric in metrics.items():
        phenotype = primary[key]
        patient, pv, _replicate, _estimator = key
        reference = full[(patient, pv)]
        if not _close(float(metric["capacity"]), float(phenotype["capacity"])):
            raise PatientSummaryError(f"metric and phenotype reconstructed capacities differ for {key}")
        if not _close(float(metric["full_capacity"]), float(reference["capacity"])):
            raise PatientSummaryError(f"metric and phenotype full capacities differ for {key}")
        wanted = abs(math.log(float(phenotype["capacity"]) / float(reference["capacity"])))
        if not _close(float(metric["log_capacity_error"]), wanted):
            raise PatientSummaryError(f"metric and phenotype log-capacity errors differ for {key}")
    return full, primary


def _median(values: Sequence[float]) -> float:
    if not values:
        raise PatientSummaryError("cannot summarize an empty sequence")
    return float(statistics.median(values))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise PatientSummaryError("cannot summarize an empty sequence")
    return float(statistics.fmean(values))


def _comparison(value: float) -> str:
    if value < -TIE_TOLERANCE:
        return "improved"
    if value > TIE_TOLERANCE:
        return "worsened"
    return "tied"


def _argmax_labels(values: Mapping[str, float]) -> tuple[str, ...]:
    maximum = max(values.values())
    tolerance = 1.0e-12 * max(1.0, abs(maximum))
    return tuple(label for label in PV_LABELS if abs(values[label] - maximum) <= tolerance)


def _build_patient_tables(
    metrics: Mapping[tuple[str, str, int, str], Mapping[str, float]],
    full: Mapping[tuple[str, str], Mapping[str, float | int]],
    phenotype: Mapping[tuple[str, str, int, str], Mapping[str, float | int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p1_rows: list[dict[str, Any]] = []
    p2_rows: list[dict[str, Any]] = []
    for patient in EXPECTED_IDS:
        p1_scores: dict[str, float] = {}
        total_area: float | None = None
        for estimator in ESTIMATORS:
            numerator = 0.0
            denominator = 0.0
            for pv in PV_LABELS:
                for replicate in range(12):
                    row = metrics[(patient, pv, replicate, estimator)]
                    numerator += row["area"] * row["brier"]
                    denominator += row["area"]
            if denominator <= 0.0:
                raise PatientSummaryError(f"patient {patient} has nonpositive pooled evaluation area")
            p1_scores[estimator] = numerator / denominator
            if total_area is None:
                total_area = denominator
            elif not _close(total_area, denominator):
                raise PatientSummaryError(f"patient {patient} methods have unequal pooled evaluation area")
        contrast = p1_scores["graph"] - p1_scores["passive"]
        p1_rows.append(
            {
                "patient_id": patient,
                "partition": _partition(patient),
                "n_pv_rotation_units": 48,
                "total_evaluation_area_mm2": total_area,
                **{f"{estimator}_brier_area_weighted": p1_scores[estimator] for estimator in ESTIMATORS},
                "graph_minus_passive_brier": contrast,
                "graph_vs_passive": _comparison(contrast),
            }
        )

        fidelity: dict[str, float] = {}
        barrier_strength: dict[str, float] = {}
        most_permissive: dict[str, tuple[str, ...]] = {}
        gap_error: dict[str, float] = {}
        widest_error: dict[str, float] = {}
        full_capacity = {pv: float(full[(patient, pv)]["capacity"]) for pv in PV_LABELS}
        full_barrier = -math.log(max(full_capacity.values()))
        full_argmax = _argmax_labels(full_capacity)
        for estimator in ESTIMATORS:
            rotation_errors: list[float] = []
            rotation_gap_errors: list[float] = []
            rotation_widest_errors: list[float] = []
            pv_medians: dict[str, float] = {}
            for replicate in range(12):
                capacity_errors: list[float] = []
                count_errors: list[float] = []
                arc_errors: list[float] = []
                for pv in PV_LABELS:
                    reconstructed = phenotype[(patient, pv, replicate, estimator)]
                    reference = full[(patient, pv)]
                    capacity_errors.append(abs(math.log(float(reconstructed["capacity"]) / float(reference["capacity"]))))
                    count_errors.append(abs(float(reconstructed["gap_count"]) - float(reference["gap_count"])))
                    arc_errors.append(abs(float(reconstructed["widest"]) - float(reference["widest"])))
                rotation_errors.append(_mean(capacity_errors))
                rotation_gap_errors.append(_mean(count_errors))
                rotation_widest_errors.append(_mean(arc_errors))
            for pv in PV_LABELS:
                pv_medians[pv] = _median(
                    [float(phenotype[(patient, pv, replicate, estimator)]["capacity"]) for replicate in range(12)]
                )
            fidelity[estimator] = _median(rotation_errors)
            barrier_strength[estimator] = -math.log(max(pv_medians.values()))
            most_permissive[estimator] = _argmax_labels(pv_medians)
            gap_error[estimator] = _median(rotation_gap_errors)
            widest_error[estimator] = _median(rotation_widest_errors)
        p2_contrast = fidelity["graph"] - fidelity["passive"]
        p2_rows.append(
            {
                "patient_id": patient,
                "partition": _partition(patient),
                "n_pv_rotation_units": 48,
                **{f"{estimator}_capacity_fidelity_error": fidelity[estimator] for estimator in ESTIMATORS},
                "graph_minus_passive_capacity_fidelity_error": p2_contrast,
                "graph_vs_passive": _comparison(p2_contrast),
                "full_barrier_strength": full_barrier,
                "passive_barrier_strength": barrier_strength["passive"],
                "graph_barrier_strength": barrier_strength["graph"],
                "passive_minus_full_barrier_strength": barrier_strength["passive"] - full_barrier,
                "graph_minus_full_barrier_strength": barrier_strength["graph"] - full_barrier,
                "full_most_permissive_pv": ";".join(full_argmax),
                "passive_most_permissive_pv": ";".join(most_permissive["passive"]),
                "graph_most_permissive_pv": ";".join(most_permissive["graph"]),
                "passive_most_permissive_pv_match": int(bool(set(full_argmax) & set(most_permissive["passive"]))),
                "graph_most_permissive_pv_match": int(bool(set(full_argmax) & set(most_permissive["graph"]))),
                "passive_gap_count_mae": gap_error["passive"],
                "graph_gap_count_mae": gap_error["graph"],
                "passive_widest_viable_arc_mae_mm": widest_error["passive"],
                "graph_widest_viable_arc_mae_mm": widest_error["graph"],
            }
        )
    return p1_rows, p2_rows


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise PatientSummaryError("cannot take a percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _bootstrap_intervals(values: Sequence[float], resamples: int) -> dict[str, list[float]]:
    if len(values) != 15:
        raise PatientSummaryError("locked-holdout bootstrap requires exactly 15 patients")
    if resamples < 1:
        raise PatientSummaryError("bootstrap resample count must be positive")
    generator = random.Random(BOOTSTRAP_SEED)
    means: list[float] = []
    medians: list[float] = []
    count = len(values)
    for _ in range(resamples):
        sample = [values[generator.randrange(count)] for _index in range(count)]
        means.append(_mean(sample))
        medians.append(_median(sample))
    means.sort()
    medians.sort()
    return {
        "mean_percentile_95_ci": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "median_percentile_95_ci": [_percentile(medians, 0.025), _percentile(medians, 0.975)],
    }


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _paired_report(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    include_individual: bool,
    bootstrap_resamples: int | None = None,
) -> dict[str, Any]:
    differences = [float(row[field]) for row in rows]
    report: dict[str, Any] = {
        **_descriptive(differences),
        "improved": sum(value < -TIE_TOLERANCE for value in differences),
        "tied": sum(abs(value) <= TIE_TOLERANCE for value in differences),
        "worsened": sum(value > TIE_TOLERANCE for value in differences),
        "tie_tolerance": TIE_TOLERANCE,
    }
    if include_individual:
        report["patient_differences"] = [
            {"patient_id": row["patient_id"], "difference": float(row[field])}
            for row in rows
        ]
    if bootstrap_resamples is not None:
        report["bootstrap"] = {
            "unit": "patient",
            "seed": BOOTSTRAP_SEED,
            "resamples": bootstrap_resamples,
            "interval_method": "percentile_linear_interpolation",
            **_bootstrap_intervals(differences, bootstrap_resamples),
        }
    return report


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = 0.5 * ((cursor + 1) + end)
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = _mean(left)
    mean_right = _mean(right)
    cross = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right, strict=True))
    square_left = sum((x - mean_left) ** 2 for x in left)
    square_right = sum((y - mean_right) ** 2 for y in right)
    denominator = math.sqrt(square_left * square_right)
    return None if denominator == 0.0 else cross / denominator


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _concordance(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = _mean(left)
    mean_right = _mean(right)
    variance_left = _mean([(value - mean_left) ** 2 for value in left])
    variance_right = _mean([(value - mean_right) ** 2 for value in right])
    covariance = _mean(
        [(x - mean_left) * (y - mean_right) for x, y in zip(left, right, strict=True)]
    )
    denominator = variance_left + variance_right + (mean_left - mean_right) ** 2
    return None if denominator == 0.0 else 2.0 * covariance / denominator


def _capacity_agreement(rows: Sequence[Mapping[str, Any]], estimator: str) -> dict[str, Any]:
    full = [float(row["full_barrier_strength"]) for row in rows]
    reconstructed = [float(row[f"{estimator}_barrier_strength"]) for row in rows]
    differences = [rec - ref for rec, ref in zip(reconstructed, full, strict=True)]
    bias = _mean(differences)
    standard_deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
    return {
        "n": len(rows),
        "spearman_rho": _spearman(full, reconstructed),
        "lin_concordance_correlation": _concordance(full, reconstructed),
        "bland_altman_bias_reconstructed_minus_full": bias,
        "bland_altman_lower_limit": bias - 1.96 * standard_deviation,
        "bland_altman_upper_limit": bias + 1.96 * standard_deviation,
        "most_permissive_pv_match_fraction": _mean(
            [float(row[f"{estimator}_most_permissive_pv_match"]) for row in rows]
        ),
        "gap_count_mae": _descriptive([float(row[f"{estimator}_gap_count_mae"]) for row in rows]),
        "widest_viable_arc_mae_mm": _descriptive(
            [float(row[f"{estimator}_widest_viable_arc_mae_mm"]) for row in rows]
        ),
    }


def _summary(
    p1_rows: Sequence[Mapping[str, Any]],
    p2_rows: Sequence[Mapping[str, Any]],
    *,
    inputs: Mapping[str, str],
    configuration_id: str,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    p1_by_id = {str(row["patient_id"]): row for row in p1_rows}
    p2_by_id = {str(row["patient_id"]): row for row in p2_rows}
    partitions = {
        "development": DEVELOPMENT_IDS,
        "locked_holdout": LOCKED_HOLDOUT_IDS,
        "all": EXPECTED_IDS,
    }
    p1_report: dict[str, Any] = {}
    p2_report: dict[str, Any] = {}
    agreement: dict[str, Any] = {}
    for partition_name, identifiers in partitions.items():
        selected_p1 = [p1_by_id[patient] for patient in identifiers]
        selected_p2 = [p2_by_id[patient] for patient in identifiers]
        p1_report[partition_name] = {
            "method_scores": {
                estimator: _descriptive(
                    [float(row[f"{estimator}_brier_area_weighted"]) for row in selected_p1]
                )
                for estimator in ESTIMATORS
            },
            "graph_minus_passive": _paired_report(
                selected_p1,
                "graph_minus_passive_brier",
                include_individual=partition_name == "locked_holdout",
                bootstrap_resamples=(bootstrap_resamples if partition_name == "locked_holdout" else None),
            ),
        }
        p2_report[partition_name] = {
            "method_scores": {
                estimator: _descriptive(
                    [float(row[f"{estimator}_capacity_fidelity_error"]) for row in selected_p2]
                )
                for estimator in ESTIMATORS
            },
            "graph_minus_passive": _paired_report(
                selected_p2,
                "graph_minus_passive_capacity_fidelity_error",
                include_individual=partition_name == "locked_holdout",
                bootstrap_resamples=(bootstrap_resamples if partition_name == "locked_holdout" else None),
            ),
        }
        agreement[partition_name] = {
            estimator: _capacity_agreement(selected_p2, estimator)
            for estimator in ("passive", "graph")
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "locked_outcome_free_patient_experiments_P1_P2",
        "configuration_id": configuration_id,
        "cohort": {
            "n_patients": 82,
            "n_development": 67,
            "n_locked_holdout": 15,
            "pv_labels": list(PV_LABELS),
            "rotations_per_pv": 12,
        },
        "definitions": {
            "P1": "within-patient area-weighted Brier pooled over 4 PVs x 12 rotations",
            "P1_primary_contrast": "graph minus passive from the same locked configuration; negative favors graph",
            "P2": "median_rotation mean_PV abs(log(reconstructed_capacity/full_capacity))",
            "P2_primary_contrast": "graph minus passive from the same locked configuration; negative favors graph",
            "barrier_strength": "-log(max_PV median_12_rotations(normalized_capacity))",
            "full_barrier_strength": "-log(max_PV full-reference normalized capacity)",
            "gap_secondary": "median_rotation mean_PV absolute reconstructed-minus-full error",
            "most_permissive_pv_match": "intersection of full and reconstructed argmax sets; tie tolerance=1e-12*max(1,abs(maximum))",
        },
        "P1": p1_report,
        "P2": p2_report,
        "secondary_capacity_and_gap_agreement": agreement,
        "input_sha256": dict(inputs),
        "outcomes_accessed": False,
    }


def _csv_text(columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _refuse_outputs(paths: Sequence[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise PatientSummaryError("output paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise PatientSummaryError("write-once output already exists: " + ", ".join(existing))


def _write_exclusive(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    except FileExistsError as exc:
        raise PatientSummaryError(f"write-once output already exists: {path}") from exc
    except OSError as exc:
        raise PatientSummaryError(f"cannot write output {path}: {exc}") from exc


def analyze(
    *,
    metrics_path: Path,
    phenotype_path: Path,
    provenance_path: Path,
    mask_manifest_path: Path,
    numerics_path: Path,
    analysis_lock_path: Path,
    expected_lock_sha256: str,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _numerics, selected = _load_locked_numerics(numerics_path)
    lock = _load_analysis_lock(analysis_lock_path, expected_lock_sha256, numerics_path)
    configuration_id = str(selected["configuration_id"])
    _load_run_provenance(
        provenance_path,
        metrics_path=metrics_path,
        phenotype_path=phenotype_path,
        mask_manifest_path=mask_manifest_path,
        numerics_path=numerics_path,
        lock=lock,
        lock_sha256=expected_lock_sha256,
        configuration_id=configuration_id,
    )
    metric_rows = _read_csv(metrics_path, RECONSTRUCTION_METRIC_COLUMNS, "reconstruction metrics")
    phenotype_rows = _read_csv(phenotype_path, PHENOTYPE_COLUMNS, "long-form phenotype")
    metrics = _validate_metric_rows(
        metric_rows,
        selected,
    )
    full, phenotype = _validate_phenotype_rows(
        phenotype_rows,
        configuration_id,
        float(lock["primary_settings"]["annulus_width_mm"]),
        float(lock["numerical_acceptance"]["capacity_relative_residual_max"]),
        metrics,
    )
    p1_rows, p2_rows = _build_patient_tables(metrics, full, phenotype)
    inputs = {
        "reconstruction_metrics": _sha256(metrics_path),
        "long_form_phenotype": _sha256(phenotype_path),
        "run_provenance": _sha256(provenance_path),
        "mask_manifest": _sha256(mask_manifest_path),
        "patient_numerics": _sha256(numerics_path),
        "analysis_lock": _sha256(analysis_lock_path),
    }
    summary = _summary(
        p1_rows,
        p2_rows,
        inputs=inputs,
        configuration_id=configuration_id,
        bootstrap_resamples=bootstrap_resamples,
    )
    return p1_rows, p2_rows, summary


def command_run(args: argparse.Namespace) -> None:
    outputs = (args.p1_output, args.p2_output, args.summary_output)
    _refuse_outputs(outputs)
    p1_rows, p2_rows, summary = analyze(
        metrics_path=args.metrics,
        phenotype_path=args.phenotype,
        provenance_path=args.run_provenance,
        mask_manifest_path=args.mask_manifest,
        numerics_path=args.patient_numerics,
        analysis_lock_path=args.analysis_lock,
        expected_lock_sha256=args.expected_lock_sha256,
    )
    rendered = (
        _csv_text(P1_COLUMNS, p1_rows),
        _csv_text(P2_COLUMNS, p2_rows),
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    for path, content in zip(outputs, rendered, strict=True):
        _write_exclusive(path, content)
    holdout_p1 = summary["P1"]["locked_holdout"]["graph_minus_passive"]
    holdout_p2 = summary["P2"]["locked_holdout"]["graph_minus_passive"]
    print(
        "wrote locked outcome-free P1/P2 summaries for 82 patients; "
        f"holdout mean contrasts: P1={holdout_p1['mean']:.8g}, P2={holdout_p2['mean']:.8g}"
    )


def _synthetic_configuration() -> dict[str, Any]:
    return {
        "configuration_id": "synthetic_locked_configuration",
        "reference_length_mm": 1.0,
        "coordinate_rule": "x_hat=x/reference_length_mm",
        "observation_confidence": 2.0,
        "probability_threshold": 0.5,
        "terminal_readout": "mean_last_k_post_step_states",
        "terminal_average_steps": 2,
        "screened": {
            "epsilon": 0.1,
            "length_scale_dimensionless": 1.0,
            "linear_method": "direct",
            "linear_tolerance": 1.0e-10,
            "linear_absolute_tolerance": 1.0e-12,
            "linear_max_iterations": 100,
        },
        "phase": {
            "mu": 1.0,
            "graph_nu": 0.5,
            "dt": 0.02,
            "nsteps": 4,
            "rho_factor": 1.0,
            "admm_tolerance": 1.0e-8,
            "admm_max_iterations": 100,
            "linear_method": "direct",
            "linear_tolerance": 1.0e-10,
            "linear_absolute_tolerance": 1.0e-12,
            "linear_max_iterations": 100,
        },
        "capacity": {
            "annulus_width_mm": 10.0,
            "outer_electrode_rule": "inner_vertex_frontier_of_geodesic_sublevel",
            "relative_residual_max": 1.0e-10,
        },
        "conductivity": {
            "formula": PRIMARY_CONDUCTIVITY_MAP,
            "full_reference_score": FULL_REFERENCE_SCORE,
            "reconstruction_score": RECONSTRUCTION_SCORE,
        },
    }


def _write_fixture(directory: Path) -> dict[str, Path | str]:
    configuration = _synthetic_configuration()
    numerics = {
        "schema_version": SCHEMA_VERSION,
        "development_candidates": [configuration],
        "selection": {
            "status": "LOCKED",
            "rule": "one_standard_error_patient_level_brier",
            "selected_without_holdout": True,
            "complexity_order": [configuration["configuration_id"]],
            "selected_configuration": configuration,
            "selection_evidence": {
                "source_draft_numerics_sha256": "1" * 64,
                "selection_report_sha256": "2" * 64,
            },
        },
    }
    numerics_path = directory / "patient_numerics.json"
    numerics_path.write_text(json.dumps(numerics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    analysis_lock = {
        "schema_version": SCHEMA_VERSION,
        "status": "LOCKED",
        "protocol": {"path": "PATIENT_EXTENSION_PROTOCOL.md", "sha256": "b" * 64},
        "source": {
            "git_commit_or_tree_hash": "d" * 40,
            "environment_lock_sha256": "c" * 64,
            "patient_input_manifest_sha256": "a" * 64,
            "patient_numerics_manifest_sha256": _sha256(numerics_path),
        },
        "outcome_blinding": {
            "clinical_manifest_unavailable_during_lock": True,
            "confirmed_by": "synthetic self-test",
            "confirmed_utc": "2025-09-03T00:00:00Z",
        },
        "primary_settings": {
            "development_size": 67,
            "locked_holdout_size": 15,
            "mask_seed": 20250903,
            "mask_replicates": 12,
            "masked_boundary_arc_fraction": 0.5,
            "annulus_width_mm": 10.0,
            "guard_width_mm": 1.5,
            "reconstruction_endpoint": "patient_area_weighted_brier",
            "phenotype_fidelity_endpoint": "patient_median_mean_abs_log_capacity_ratio",
            "recurrence_endpoint": "locked_holdout_delta_log_loss_m1_minus_m0",
        },
        "frozen_phenotype": {
            "name": "B_i",
            "estimator": "direct_graph_reconstruction_from_primary_50_percent_masks",
            "capacity_per_rotation": "normalized_capacity",
            "within_pv_rotation_aggregation": "median",
            "across_pv_aggregation": "maximum",
            "operation_order": [
                "normalized_capacity_per_rotation_per_pv",
                "median_over_12_rotations_within_each_pv",
                "maximum_over_pvs",
                "negative_log",
            ],
            "definition": "-log(max_pv(median_12_rotations(normalized_capacity)))",
            "conductivity_map": PRIMARY_CONDUCTIVITY_MAP,
            "full_reference_score": FULL_REFERENCE_SCORE,
            "reconstruction_score": RECONSTRUCTION_SCORE,
            "probability_clipping_only": True,
            "log_offset": 0.0,
            "strict_positive_capacity_required": True,
            "higher_means": "stronger_modeled_barrier",
        },
        "numerical_acceptance": {
            "capacity_relative_residual_max": 1.0e-10,
            "capacity_two_finest_relative_change_max": 0.05,
            "whole_surface_valid_area_fraction_min": 0.90,
            "pv_annulus_valid_area_fraction_min": 0.95,
        },
        "locked_outputs": {
            "directory": "synthetic_write_once_directory",
            "created_utc": "2025-09-03T00:00:00Z",
            "manifest_sha256": "e" * 64,
        },
    }
    lock_path = directory / "analysis_lock.json"
    lock_path.write_text(json.dumps(analysis_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock_sha = _sha256(lock_path)

    method_brier = {
        "prevalence": 0.14,
        "nearest_neighbour": 0.12,
        "screened": 0.10,
        "passive": 0.08,
        "graph": 0.06,
    }
    capacity_multiplier = {
        "prevalence": 1.10,
        "nearest_neighbour": 1.08,
        "screened": 1.06,
        "passive": 1.04,
        "graph": 1.02,
    }
    metric_rows: list[dict[str, Any]] = []
    phenotype_rows: list[dict[str, Any]] = []
    for patient_index, patient in enumerate(EXPECTED_IDS):
        partition = _partition(patient)
        for pv_index, pv in enumerate(PV_LABELS):
            full_capacity = 0.02 + 0.001 * pv_index + 0.000001 * patient_index
            full_widest = 3.0 + pv_index
            full_gaps = pv_index % 3
            phenotype_rows.append(
                {
                    "patient_id": patient,
                    "partition": partition,
                    "configuration_id": configuration["configuration_id"],
                    "estimator": "full_imaging_reference",
                    "replicate": -1,
                    "mask_variant": "full",
                    "pv_label": pv,
                    "annulus_width_mm": 10.0,
                    "conductivity_map": PRIMARY_CONDUCTIVITY_MAP,
                    "conductivity_score_source": FULL_REFERENCE_SCORE,
                    "normalised_capacity": full_capacity,
                    "capacity_relative_residual": 1.0e-12,
                    "widest_viable_arc_mm": full_widest,
                    "viable_arc_count": full_gaps,
                    "minimum_barrier_score": -0.5,
                    "phenotype_complete": 1,
                }
            )
            for replicate in range(12):
                area = 2.0 + 0.1 * pv_index + 0.01 * replicate
                for method_index, estimator in enumerate(ESTIMATORS):
                    reconstructed_capacity = full_capacity * capacity_multiplier[estimator] * (1.0 + 0.0001 * replicate)
                    brier = method_brier[estimator] + 0.00001 * patient_index + 0.00002 * replicate
                    metric_rows.append(
                        {
                            "patient_id": patient,
                            "partition": partition,
                            "pv_label": pv,
                            "replicate": replicate,
                            "mask_variant": "primary",
                            "sector_fraction": 0.5,
                            "band_width_mm": 10.0,
                            "evaluation_area_mm2": area,
                            "configuration_id": configuration["configuration_id"],
                            "estimator": estimator,
                            "rmse_area_weighted": math.sqrt(brier),
                            "mae_area_weighted": min(1.0, brier * 1.5),
                            "brier_area_weighted": brier,
                            "dice_barrier": 0.75,
                            "balanced_accuracy": 0.8,
                            "boundary_distance_mm": 1.0,
                            "topology_match": 1,
                            "normalised_capacity": reconstructed_capacity,
                            "full_normalised_capacity": full_capacity,
                            "capacity_relative_error": abs(reconstructed_capacity - full_capacity) / full_capacity,
                            "absolute_log_capacity_ratio": abs(math.log(reconstructed_capacity / full_capacity)),
                            "solver_converged": 1,
                            "solver_residual": (
                                0.0
                                if estimator in {"prevalence", "nearest_neighbour"}
                                else 1.0e-12
                            ),
                        }
                    )
                    phenotype_rows.append(
                        {
                            "patient_id": patient,
                            "partition": partition,
                            "configuration_id": configuration["configuration_id"],
                            "estimator": estimator,
                            "replicate": replicate,
                            "mask_variant": "primary",
                            "pv_label": pv,
                            "annulus_width_mm": 10.0,
                            "conductivity_map": PRIMARY_CONDUCTIVITY_MAP,
                            "conductivity_score_source": RECONSTRUCTION_SCORE,
                            "normalised_capacity": reconstructed_capacity,
                            "capacity_relative_residual": 1.0e-12,
                            "widest_viable_arc_mm": full_widest + 0.1 * method_index,
                            "viable_arc_count": full_gaps + (1 if estimator in {"prevalence", "nearest_neighbour"} else 0),
                            "minimum_barrier_score": -0.5 + 0.01 * method_index,
                            "phenotype_complete": 1,
                        }
                    )
    metrics_path = directory / "metrics.csv"
    phenotype_path = directory / "phenotype.csv"
    mask_manifest_path = directory / "patient_mask_manifest.csv"
    metrics_path.write_text(_csv_text(RECONSTRUCTION_METRIC_COLUMNS, metric_rows), encoding="utf-8")
    phenotype_path.write_text(_csv_text(PHENOTYPE_COLUMNS, phenotype_rows), encoding="utf-8")
    mask_manifest_path.write_text(
        "patient_id,pv_label,replicate,mask_sha256\n"
        "ID001,LSPV,0," + "9" * 64 + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "partition": "all",
        "n_patients": 82,
        "configuration_id": configuration["configuration_id"],
        "configuration_mode": "locked_selection",
        "patient_input_manifest_sha256": analysis_lock["source"]["patient_input_manifest_sha256"],
        "patient_numerics_manifest_sha256": _sha256(numerics_path),
        "analysis_lock_sha256": lock_sha,
        "mask_manifest_sha256": _sha256(mask_manifest_path),
        "reconstruction_metrics_sha256": _sha256(metrics_path),
        "phenotype_table_sha256": _sha256(phenotype_path),
        "outcomes_accessed": False,
    }
    provenance_path = directory / "run_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "metrics": metrics_path,
        "phenotype": phenotype_path,
        "run_provenance": provenance_path,
        "mask_manifest": mask_manifest_path,
        "patient_numerics": numerics_path,
        "analysis_lock": lock_path,
        "expected_lock_sha256": lock_sha,
    }


def _expect_failure(callable_object: Any, phrase: str) -> None:
    try:
        callable_object()
    except PatientSummaryError:
        return
    raise PatientSummaryError(f"synthetic negative check did not reject {phrase}")


def command_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="patient_summary_self_test_") as name:
        directory = Path(name)
        fixture = _write_fixture(directory)
        paths = {key: value for key, value in fixture.items() if isinstance(value, Path)}
        lock_sha = str(fixture["expected_lock_sha256"])
        p1, p2, summary = analyze(
            metrics_path=paths["metrics"],
            phenotype_path=paths["phenotype"],
            provenance_path=paths["run_provenance"],
            mask_manifest_path=paths["mask_manifest"],
            numerics_path=paths["patient_numerics"],
            analysis_lock_path=paths["analysis_lock"],
            expected_lock_sha256=lock_sha,
            bootstrap_resamples=2_000,
        )
        if len(p1) != 82 or len(p2) != 82 or summary.get("outcomes_accessed") is not False:
            raise PatientSummaryError("synthetic complete fixture did not produce 82 outcome-free rows")
        holdout = summary["P1"]["locked_holdout"]["graph_minus_passive"]
        if holdout["n"] != 15 or holdout["improved"] != 15 or len(holdout["patient_differences"]) != 15:
            raise PatientSummaryError("synthetic holdout report is incomplete")

        wrong_mask_path = directory / "wrong_patient_mask_manifest.csv"
        wrong_mask_path.write_text(
            paths["mask_manifest"].read_text(encoding="utf-8")
            + "ID002,LIPV,1," + "8" * 64 + "\n",
            encoding="utf-8",
        )
        _expect_failure(
            lambda: analyze(
                metrics_path=paths["metrics"],
                phenotype_path=paths["phenotype"],
                provenance_path=paths["run_provenance"],
                mask_manifest_path=wrong_mask_path,
                numerics_path=paths["patient_numerics"],
                analysis_lock_path=paths["analysis_lock"],
                expected_lock_sha256=lock_sha,
                bootstrap_resamples=10,
            ),
            "wrong mask-manifest file",
        )

        bad_provenance = json.loads(paths["run_provenance"].read_text(encoding="utf-8"))
        bad_provenance["outcomes_accessed"] = True
        bad_provenance_path = directory / "bad_provenance.json"
        bad_provenance_path.write_text(json.dumps(bad_provenance, sort_keys=True) + "\n", encoding="utf-8")
        _expect_failure(
            lambda: analyze(
                metrics_path=paths["metrics"],
                phenotype_path=paths["phenotype"],
                provenance_path=bad_provenance_path,
                mask_manifest_path=paths["mask_manifest"],
                numerics_path=paths["patient_numerics"],
                analysis_lock_path=paths["analysis_lock"],
                expected_lock_sha256=lock_sha,
                bootstrap_resamples=10,
            ),
            "outcomes_accessed=true provenance",
        )

        numeric_false_provenance = json.loads(
            paths["run_provenance"].read_text(encoding="utf-8")
        )
        numeric_false_provenance["outcomes_accessed"] = 0
        numeric_false_path = directory / "numeric_false_provenance.json"
        numeric_false_path.write_text(
            json.dumps(numeric_false_provenance, sort_keys=True) + "\n", encoding="utf-8"
        )
        _expect_failure(
            lambda: analyze(
                metrics_path=paths["metrics"],
                phenotype_path=paths["phenotype"],
                provenance_path=numeric_false_path,
                mask_manifest_path=paths["mask_manifest"],
                numerics_path=paths["patient_numerics"],
                analysis_lock_path=paths["analysis_lock"],
                expected_lock_sha256=lock_sha,
                bootstrap_resamples=10,
            ),
            "numeric outcomes_accessed value",
        )

        bad_hash_provenance = json.loads(paths["run_provenance"].read_text(encoding="utf-8"))
        bad_hash_provenance["phenotype_table_sha256"] = "0" * 64
        bad_hash_path = directory / "bad_hash_provenance.json"
        bad_hash_path.write_text(
            json.dumps(bad_hash_provenance, sort_keys=True) + "\n", encoding="utf-8"
        )
        _expect_failure(
            lambda: analyze(
                metrics_path=paths["metrics"],
                phenotype_path=paths["phenotype"],
                provenance_path=bad_hash_path,
                mask_manifest_path=paths["mask_manifest"],
                numerics_path=paths["patient_numerics"],
                analysis_lock_path=paths["analysis_lock"],
                expected_lock_sha256=lock_sha,
                bootstrap_resamples=10,
            ),
            "incorrect phenotype checksum",
        )

        metric_rows = _read_csv(paths["metrics"], RECONSTRUCTION_METRIC_COLUMNS, "synthetic metrics")
        incomplete_path = directory / "incomplete_metrics.csv"
        incomplete_path.write_text(
            _csv_text(RECONSTRUCTION_METRIC_COLUMNS, metric_rows[:-1]), encoding="utf-8"
        )
        incomplete_provenance = json.loads(paths["run_provenance"].read_text(encoding="utf-8"))
        incomplete_provenance["reconstruction_metrics_sha256"] = _sha256(incomplete_path)
        incomplete_provenance_path = directory / "incomplete_provenance.json"
        incomplete_provenance_path.write_text(
            json.dumps(incomplete_provenance, sort_keys=True) + "\n", encoding="utf-8"
        )
        _expect_failure(
            lambda: analyze(
                metrics_path=incomplete_path,
                phenotype_path=paths["phenotype"],
                provenance_path=incomplete_provenance_path,
                mask_manifest_path=paths["mask_manifest"],
                numerics_path=paths["patient_numerics"],
                analysis_lock_path=paths["analysis_lock"],
                expected_lock_sha256=lock_sha,
                bootstrap_resamples=10,
            ),
            "incomplete 82 x 4 x 12 x 5 metric product",
        )

        metric_lines = paths["metrics"].read_text(encoding="utf-8").splitlines()
        metric_lines[1] += ",unexpected_trailing_cell"
        trailing_path = directory / "trailing_metrics.csv"
        trailing_path.write_text("\n".join(metric_lines) + "\n", encoding="utf-8")
        trailing_provenance = json.loads(
            paths["run_provenance"].read_text(encoding="utf-8")
        )
        trailing_provenance["reconstruction_metrics_sha256"] = _sha256(trailing_path)
        trailing_provenance_path = directory / "trailing_provenance.json"
        trailing_provenance_path.write_text(
            json.dumps(trailing_provenance, sort_keys=True) + "\n", encoding="utf-8"
        )
        _expect_failure(
            lambda: analyze(
                metrics_path=trailing_path,
                phenotype_path=paths["phenotype"],
                provenance_path=trailing_provenance_path,
                mask_manifest_path=paths["mask_manifest"],
                numerics_path=paths["patient_numerics"],
                analysis_lock_path=paths["analysis_lock"],
                expected_lock_sha256=lock_sha,
                bootstrap_resamples=10,
            ),
            "trailing CSV cells",
        )

        p1_path = directory / "p1.csv"
        p2_path = directory / "p2.csv"
        summary_path = directory / "summary.json"
        for path, text in (
            (p1_path, _csv_text(P1_COLUMNS, p1)),
            (p2_path, _csv_text(P2_COLUMNS, p2)),
            (summary_path, json.dumps(summary, allow_nan=False)),
        ):
            _write_exclusive(path, text)
        _expect_failure(lambda: _refuse_outputs((p1_path, p2_path, summary_path)), "existing outputs")
    print(
        "patient P1/P2 summary self-test passed: complete 82-case fixture, "
        "mask-lineage/outcome-blinding/checksum provenance failures, negative "
        "completeness, trailing-cell rejection, and write-once checks"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce locked, outcome-free P1/P2 patient summaries."
    )
    parser.add_argument("--self-test", action="store_true", help="run the dependency-free synthetic checks")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--phenotype", type=Path)
    parser.add_argument("--run-provenance", type=Path)
    parser.add_argument("--mask-manifest", type=Path)
    parser.add_argument("--patient-numerics", type=Path)
    parser.add_argument("--analysis-lock", type=Path)
    parser.add_argument("--expected-lock-sha256")
    parser.add_argument("--p1-output", type=Path)
    parser.add_argument("--p2-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_test:
        supplied = [
            args.metrics,
            args.phenotype,
            args.run_provenance,
            args.mask_manifest,
            args.patient_numerics,
            args.analysis_lock,
            args.expected_lock_sha256,
            args.p1_output,
            args.p2_output,
            args.summary_output,
        ]
        if any(value is not None for value in supplied):
            parser.error("--self-test cannot be combined with production inputs or outputs")
        try:
            command_self_test()
        except PatientSummaryError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        return 0
    required = {
        "--metrics": args.metrics,
        "--phenotype": args.phenotype,
        "--run-provenance": args.run_provenance,
        "--mask-manifest": args.mask_manifest,
        "--patient-numerics": args.patient_numerics,
        "--analysis-lock": args.analysis_lock,
        "--expected-lock-sha256": args.expected_lock_sha256,
        "--p1-output": args.p1_output,
        "--p2-output": args.p2_output,
        "--summary-output": args.summary_output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("production mode requires " + ", ".join(missing))
    try:
        command_run(args)
    except PatientSummaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
