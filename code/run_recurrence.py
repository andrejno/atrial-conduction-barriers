"""Fail-closed recurrence association analysis for an authorised linked cohort.

This program never downloads, guesses, or fabricates outcomes.  It accepts two
explicit, ID-linked tables only after checking an authorisation record, an
outcome-blind feature lock, and SHA-256 hashes for all three inputs.  The primary
binary analysis fits Jeffreys-penalised (Firth) logistic models in the published
development partition and evaluates them once in the unchanged holdout.  A
development-only leave-one-patient-out estimate is secondary.  The optional
time-to-event analysis applies the same partition to a post-MRI landmark Cox
model with a fixed ridge penalty and Breslow handling of ties.

For the named 82-case Dryad extension, all 82 locked phenotypes are mandatory.
A structural QC loss stops this clinical analysis instead of shrinking the
already small 15-person holdout.

The outputs quantify fixed-holdout and secondary development-only association;
they are not causal estimates or external validation.

Example
-------
python code/run_recurrence.py \
    --outcomes /secure/path/recurrence_outcomes.csv \
    --phenotypes /secure/path/pvi_phenotypes.csv \
    --authorization /secure/path/recurrence_authorization.json \
    --feature-lock /secure/path/recurrence_feature_lock.json \
    --structural-analysis-lock /secure/path/analysis_lock.manifest.json \
    --phenotype-provenance /secure/path/combined_phenotype.provenance.json \
    --analysis both \
    --output-dir /secure/path/recurrence_results \
    --acknowledge-authorized-linkage

Use ``--hash-file PATH`` to calculate a SHA-256 value for a prepared input and
``--self-test`` to exercise both analysis paths using temporary synthetic data.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logsumexp
from scipy.stats import rankdata


LOCKED_BINARY_ESTIMATOR = "firth_jeffreys_penalized_logistic"
LOCKED_SURVIVAL_ESTIMATOR = "ridge_cox_breslow"
LOCKED_PRIMARY_EVALUATION = "published_fixed_holdout"
LOCKED_CROSS_FITTING = "development_only_leave_one_patient_out"
LOCKED_STANDARDIZATION = "training_data_only"
ALLOWED_TRANSFORMS = {
    "identity",
    "log1p_nonnegative",
    "log_positive",
    "sqrt_nonnegative",
}
FORBIDDEN_FEATURE_TOKENS = {
    "recurrence",
    "outcome",
    "event",
    "followup",
    "follow_up",
    "censor",
    "survival",
}
DATE_COLUMNS = {
    "ablation_date",
    "post_ablation_mri_date",
    "followup_end_date",
    "recurrence_date",
}
BASE_OUTCOME_COLUMNS = {"patient_id", "recurrence_2y"}


class RecurrencePipelineError(RuntimeError):
    """Base class for a controlled, fail-closed termination."""


class AuthorizationError(RecurrencePipelineError):
    """The linkage or input files have not been explicitly authorised."""


class DataValidationError(RecurrencePipelineError):
    """An input does not satisfy the locked schema or chronology."""


class ModelFitError(RecurrencePipelineError):
    """A prespecified model could not be estimated reliably."""


STATIONARITY_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    role: str
    transform: str
    units: str
    direction: str
    description: str


@dataclass(frozen=True)
class AnalysisConfig:
    outcomes: Path
    phenotypes: Path
    authorization: Path
    feature_lock: Path
    structural_analysis_lock: Path
    phenotype_provenance: Path
    analysis: str
    output_dir: Path
    acknowledge_authorized_linkage: bool


def _require(condition: bool, message: str, exc: type[RecurrencePipelineError] = DataValidationError) -> None:
    if not condition:
        raise exc(message)


def _is_json_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sha256_file(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    _require(path.is_file(), f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def _parse_utc(value: object, label: str) -> datetime:
    _require(isinstance(value, str) and value.strip(), f"{label} must be a nonempty UTC timestamp")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise DataValidationError(f"{label} is not an ISO-8601 timestamp") from error
    _require(parsed.tzinfo is not None, f"{label} must include a UTC offset")
    parsed = parsed.astimezone(timezone.utc)
    _require(parsed <= datetime.now(timezone.utc), f"{label} cannot be in the future")
    return parsed


def _require_keys(record: Mapping[str, object], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    missing = expected.difference(record)
    _require(not missing, f"{label} is missing fields: {sorted(missing)}")
    extra = set(record).difference(expected)
    _require(not extra, f"{label} has unsupported fields: {sorted(extra)}")


def _require_keys_subset(record: Mapping[str, object], keys: Iterable[str], label: str) -> None:
    missing = set(keys).difference(record)
    _require(not missing, f"{label} is missing fields: {sorted(missing)}")


def _contains_replace_marker(value: object) -> bool:
    if isinstance(value, str):
        return "REPLACE" in value.upper()
    if isinstance(value, Mapping):
        return any(_contains_replace_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_replace_marker(item) for item in value)
    return False


def validate_feature_lock(lock: Mapping[str, object]) -> tuple[list[FeatureSpec], Mapping[str, object]]:
    """Validate and return the immutable, outcome-blind feature specification."""
    _require_keys(
        lock,
        {
            "lock_version",
            "locked",
            "locked_at_utc",
            "outcome_information_used",
            "cohort_name",
            "source_doi",
            "structural_analysis_lock_sha256",
            "phenotype_provenance_sha256",
            "phenotype_table_sha256",
            "expected_patient_count",
            "id_column",
            "partition",
            "phenotype_definition",
            "features",
            "analysis",
        },
        "feature lock",
    )
    _require(lock["lock_version"] == "1.0", "feature lock lock_version must be 1.0")
    _require(lock["locked"] is True, "feature lock must have locked=true")
    _require(lock["outcome_information_used"] is False, "feature lock must attest outcome_information_used=false")
    _parse_utc(lock["locked_at_utc"], "feature lock locked_at_utc")
    _require(lock["id_column"] == "patient_id", "the only supported linkage key is patient_id")
    _require(isinstance(lock["cohort_name"], str) and lock["cohort_name"].strip(), "cohort_name is empty")
    _require(isinstance(lock["source_doi"], str) and lock["source_doi"].strip(), "source_doi is empty")
    for field in (
        "structural_analysis_lock_sha256",
        "phenotype_provenance_sha256",
        "phenotype_table_sha256",
    ):
        value = lock[field]
        _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                 f"{field} must be a lowercase SHA-256 digest")
        _require(value != "0" * 64, f"{field} is still an unset template value")
    expected_n = lock["expected_patient_count"]
    _require(_is_json_integer(expected_n) and expected_n >= 20,
             "expected_patient_count must be an integer of at least 20")

    partition = lock["partition"]
    _require(isinstance(partition, dict), "feature-lock partition field must be an object")
    _require_keys(partition, {"source", "development_ids", "holdout_ids"}, "feature-lock partition")
    _require(isinstance(partition["source"], str) and partition["source"].strip(), "partition source is empty")
    for field, minimum in (("development_ids", 20), ("holdout_ids", 10)):
        identifiers = partition[field]
        _require(isinstance(identifiers, list) and len(identifiers) >= minimum,
                 f"partition {field} must contain at least {minimum} IDs")
        _require(all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value)
                     for value in identifiers), f"partition {field} contains an invalid ID")
        _require(len(identifiers) == len(set(identifiers)), f"partition {field} contains duplicate IDs")
    development_ids = set(partition["development_ids"])
    holdout_ids = set(partition["holdout_ids"])
    _require(development_ids.isdisjoint(holdout_ids), "development and holdout IDs overlap")
    _require(len(development_ids | holdout_ids) == expected_n,
             "partition union does not equal expected_patient_count")
    if lock["source_doi"] == "10.5061/dryad.kkwh70sg0":
        published_development = {f"ID{value:03d}" for value in range(21, 88)}
        published_holdout = {f"ID{value:03d}" for value in range(1, 16)}
        _require(development_ids == published_development and holdout_ids == published_holdout,
                 "Dryad cohort must retain the published 67/15 development/holdout split")

    phenotype_definition = lock["phenotype_definition"]
    _require(isinstance(phenotype_definition, dict), "phenotype_definition must be an object")
    _require_keys(
        phenotype_definition,
        {
            "primary_feature",
            "symbol",
            "estimator",
            "capacity_per_rotation",
            "conductivity_map",
            "full_reference_score",
            "reconstruction_score",
            "probability_clipping_only",
            "within_pv_rotation_aggregation",
            "across_pv_aggregation",
            "operation_order",
            "definition",
            "log_offset",
            "strict_positive_capacity_required",
            "higher_means",
            "outcome_blind",
        },
        "phenotype_definition",
    )
    _require(phenotype_definition["outcome_blind"] is True,
             "phenotype_definition.outcome_blind must be true")
    _require(isinstance(phenotype_definition["primary_feature"], str),
             "phenotype_definition.primary_feature must be a feature name")
    _require(isinstance(phenotype_definition["definition"], str) and phenotype_definition["definition"].strip(),
             "phenotype_definition.definition is empty")
    _require(isinstance(phenotype_definition["operation_order"], list)
             and len(phenotype_definition["operation_order"]) >= 3,
             "phenotype_definition.operation_order is incomplete")
    _require(phenotype_definition["strict_positive_capacity_required"] is True,
             "phenotype_definition must require strictly positive capacities")
    _require(phenotype_definition["probability_clipping_only"] is True,
             "phenotype_definition must restrict clipping to probability scoring")
    _require(_is_json_number(phenotype_definition["log_offset"])
             and phenotype_definition["log_offset"] == 0,
             "phenotype_definition.log_offset must be exactly zero")

    raw_features = lock["features"]
    _require(isinstance(raw_features, list) and raw_features, "feature lock must define at least one feature")
    features: list[FeatureSpec] = []
    for position, raw in enumerate(raw_features):
        _require(isinstance(raw, dict), f"feature {position} is not an object")
        _require_keys(raw, {"name", "role", "transform", "units", "direction", "description"}, f"feature {position}")
        name = raw["name"]
        _require(isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is not None,
                 f"feature {position} has an invalid name")
        lowered = name.lower()
        _require(not any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS),
                 f"feature {name} has an outcome-like name and is not permitted")
        role = raw["role"]
        _require(role in {"phenotype", "clinical"}, f"feature {name} has unsupported role {role!r}")
        transform = raw["transform"]
        _require(transform in ALLOWED_TRANSFORMS, f"feature {name} has unsupported transform {transform!r}")
        for field in ("units", "direction", "description"):
            _require(isinstance(raw[field], str) and raw[field].strip(), f"feature {name}: {field} is empty")
        features.append(FeatureSpec(name, role, transform, raw["units"], raw["direction"], raw["description"]))

    names = [feature.name for feature in features]
    _require(len(names) == len(set(names)), "feature names are not unique")
    phenotype_count = sum(feature.role == "phenotype" for feature in features)
    clinical_count = sum(feature.role == "clinical" for feature in features)
    _require(1 <= phenotype_count <= 3, "lock must contain one to three phenotype features")
    _require(clinical_count <= 2, "lock may contain at most two clinical adjustment features")
    phenotype_names = [feature.name for feature in features if feature.role == "phenotype"]
    clinical_names = {feature.name for feature in features if feature.role == "clinical"}
    _require(phenotype_definition["primary_feature"] in phenotype_names,
             "phenotype_definition.primary_feature is not a locked phenotype feature")
    if lock["source_doi"] == "10.5061/dryad.kkwh70sg0":
        _require(phenotype_names == ["pvi_barrier_strength"],
                 "Dryad primary analysis must use only the frozen continuous pvi_barrier_strength phenotype")
        expected_phenotype_definition = {
            "primary_feature": "pvi_barrier_strength",
            "symbol": "B_i",
            "estimator": "direct_graph_reconstruction_from_primary_50_percent_masks",
            "capacity_per_rotation": "normalized_capacity",
            "conductivity_map": "eta(s)=1e-3+(1-1e-3)/(1+exp(8s))",
            "full_reference_score": "signed_q_2b_minus_1",
            "reconstruction_score": "unclipped_u",
            "probability_clipping_only": True,
            "within_pv_rotation_aggregation": "median",
            "across_pv_aggregation": "maximum",
            "operation_order": [
                "normalized_capacity_per_rotation_per_pv",
                "median_over_12_rotations_within_each_pv",
                "maximum_over_pvs",
                "negative_log",
            ],
            "definition": "-log(max_pv(median_12_rotations(normalized_capacity)))",
            "log_offset": 0.0,
            "strict_positive_capacity_required": True,
            "higher_means": "stronger_modeled_barrier",
            "outcome_blind": True,
        }
        _require(dict(phenotype_definition) == expected_phenotype_definition,
                 "Dryad phenotype_definition does not exactly match the locked patient protocol")
        allowed_clinical = {"post_ablation_lavi_ml_m2", "post_ablation_scar_fraction"}
        _require(clinical_names in (set(), allowed_clinical),
                 "Dryad clinical baseline must be absent or exactly post-ablation LAVI and scar fraction")

    analysis = lock["analysis"]
    _require(isinstance(analysis, dict), "feature-lock analysis field must be an object")
    _require_keys(
        analysis,
        {
            "binary_horizon_days",
            "blanking_period_days",
            "primary_evaluation",
            "cross_fitting",
            "standardization",
            "binary_estimator",
            "survival_estimator",
            "cox_ridge",
            "bootstrap_replicates",
            "bootstrap_seed",
            "permutation_replicates",
            "permutation_seed",
            "complete_cohort_required",
            "missing_predictors",
            "max_total_predictors",
        },
        "feature-lock analysis",
    )
    _require(analysis["binary_horizon_days"] == 730, "binary_horizon_days must be locked at 730")
    _require(_is_json_integer(analysis["blanking_period_days"])
             and 0 <= analysis["blanking_period_days"] <= 180,
             "blanking_period_days must be an integer between 0 and 180")
    _require(analysis["primary_evaluation"] == LOCKED_PRIMARY_EVALUATION,
             f"primary_evaluation must be {LOCKED_PRIMARY_EVALUATION}")
    _require(analysis["cross_fitting"] == LOCKED_CROSS_FITTING,
             f"cross_fitting must be {LOCKED_CROSS_FITTING}")
    _require(analysis["standardization"] == LOCKED_STANDARDIZATION,
             f"standardization must be {LOCKED_STANDARDIZATION}")
    _require(analysis["binary_estimator"] == LOCKED_BINARY_ESTIMATOR,
             f"binary_estimator must be {LOCKED_BINARY_ESTIMATOR}")
    _require(analysis["survival_estimator"] == LOCKED_SURVIVAL_ESTIMATOR,
             f"survival_estimator must be {LOCKED_SURVIVAL_ESTIMATOR}")
    _require(analysis["missing_predictors"] == "fail", "missing_predictors must be locked to fail")
    max_predictors = analysis["max_total_predictors"]
    _require(_is_json_integer(max_predictors) and 1 <= max_predictors <= 4,
             "max_total_predictors must be an integer from one to four")
    _require(len(features) <= max_predictors, "locked feature count exceeds max_total_predictors")
    cox_ridge = analysis["cox_ridge"]
    _require(_is_json_number(cox_ridge) and np.isfinite(cox_ridge) and 0.0 < cox_ridge <= 10.0,
             "cox_ridge must be fixed in (0,10]")
    bootstrap_replicates = analysis["bootstrap_replicates"]
    _require(_is_json_integer(bootstrap_replicates) and 200 <= bootstrap_replicates <= 20000,
             "bootstrap_replicates must be an integer from 200 to 20000")
    _require(_is_json_integer(analysis["bootstrap_seed"]) and analysis["bootstrap_seed"] >= 0,
             "bootstrap_seed must be a nonnegative integer")
    permutation_replicates = analysis["permutation_replicates"]
    _require(_is_json_integer(permutation_replicates) and 20 <= permutation_replicates <= 10000,
             "permutation_replicates must be an integer from 20 to 10000")
    _require(_is_json_integer(analysis["permutation_seed"]) and analysis["permutation_seed"] >= 0,
             "permutation_seed must be a nonnegative integer")
    _require(analysis["complete_cohort_required"] is True,
             "complete_cohort_required must be true for the locked clinical analysis")
    if lock["source_doi"] == "10.5061/dryad.kkwh70sg0":
        _require(analysis["blanking_period_days"] == 90,
                 "Dryad recurrence endpoint requires the published 90-day blanking period")
    return features, analysis


def validate_authorization(
    authorization: Mapping[str, object],
    lock: Mapping[str, object],
    lock_path: Path,
    outcomes_path: Path,
    phenotypes_path: Path,
    analysis_mode: str,
    acknowledged: bool,
) -> None:
    """Validate linkage permission and exact hashes before parsing either table."""
    _require(acknowledged, "explicit --acknowledge-authorized-linkage is required", AuthorizationError)
    _require_keys(
        authorization,
        {
            "authorization_version",
            "data_use_authorized",
            "id_linkage_authorized",
            "deidentified_or_pseudonymized",
            "outcomes_publicly_available",
            "recurrence_definition_verified",
            "feature_definitions_locked_before_linkage",
            "recurrence_date_blank_means_censored",
            "authorized_by",
            "authorized_at_utc",
            "outcome_linkage_at_utc",
            "data_use_basis",
            "outcome_definition",
            "minimum_duration_seconds",
            "blanking_period_days",
            "binary_horizon_days",
            "eligible_rhythms",
            "cohort_name",
            "source_doi",
            "id_linkage_key",
            "authorized_analyses",
            "feature_lock_sha256",
            "outcomes_sha256",
            "phenotypes_sha256",
        },
        "authorization record",
    )
    _require(authorization["authorization_version"] == "1.0",
             "authorization_version must be 1.0", AuthorizationError)
    required_true = {
        "data_use_authorized",
        "id_linkage_authorized",
        "deidentified_or_pseudonymized",
        "recurrence_definition_verified",
        "feature_definitions_locked_before_linkage",
        "recurrence_date_blank_means_censored",
    }
    for field in required_true:
        _require(authorization[field] is True, f"authorization field {field} must be true", AuthorizationError)
    _require(authorization["outcomes_publicly_available"] is False,
             "outcomes_publicly_available must be false; public meshes do not include the ID linkage",
             AuthorizationError)
    for field in ("authorized_by", "data_use_basis", "outcome_definition"):
        _require(isinstance(authorization[field], str) and len(authorization[field].strip()) >= 8,
                 f"authorization field {field} is incomplete", AuthorizationError)
    _require(_is_json_integer(authorization["minimum_duration_seconds"])
             and authorization["minimum_duration_seconds"] == 30,
             "authorized endpoint minimum_duration_seconds must be 30", AuthorizationError)
    _require(_is_json_integer(authorization["blanking_period_days"])
             and authorization["blanking_period_days"] == lock["analysis"]["blanking_period_days"],
             "authorization blanking period differs from the feature lock", AuthorizationError)
    _require(_is_json_integer(authorization["binary_horizon_days"])
             and authorization["binary_horizon_days"] == lock["analysis"]["binary_horizon_days"],
             "authorization binary horizon differs from the feature lock", AuthorizationError)
    _require(authorization["eligible_rhythms"] == ["atrial_fibrillation", "atrial_flutter"],
             "eligible_rhythms must be exactly atrial_fibrillation and atrial_flutter", AuthorizationError)
    _require(authorization["cohort_name"] == lock["cohort_name"],
             "authorization and feature lock name different cohorts", AuthorizationError)
    _require(authorization["source_doi"] == lock["source_doi"],
             "authorization and feature lock have different source DOI values", AuthorizationError)
    _require(authorization["phenotypes_sha256"] == lock["phenotype_table_sha256"],
             "authorization phenotype hash differs from the pre-linkage feature lock", AuthorizationError)
    _require(authorization["id_linkage_key"] == "patient_id", "id_linkage_key must be patient_id", AuthorizationError)

    scopes = authorization["authorized_analyses"]
    _require(isinstance(scopes, list), "authorized_analyses must be a list", AuthorizationError)
    _require(all(isinstance(scope, str) for scope in scopes),
             "authorized_analyses must contain strings", AuthorizationError)
    _require(len(scopes) == len(set(scopes)),
             "authorized_analyses contains duplicates", AuthorizationError)
    allowed_scopes = {"binary_2y", "post_mri_landmark"}
    _require(set(scopes).issubset(allowed_scopes),
             "authorized_analyses contains an unknown scope", AuthorizationError)
    required_scopes = {"binary_2y"} if analysis_mode == "binary" else {"post_mri_landmark"}
    if analysis_mode == "both":
        required_scopes = {"binary_2y", "post_mri_landmark"}
    _require(required_scopes.issubset(set(scopes)),
             f"authorization does not cover analyses: {sorted(required_scopes.difference(scopes))}", AuthorizationError)

    lock_time = _parse_utc(lock["locked_at_utc"], "feature lock locked_at_utc")
    authorization_time = _parse_utc(authorization["authorized_at_utc"], "authorization authorized_at_utc")
    linkage_time = _parse_utc(authorization["outcome_linkage_at_utc"], "authorization outcome_linkage_at_utc")
    _require(authorization_time <= linkage_time, "authorization must precede outcome linkage", AuthorizationError)
    _require(lock_time < linkage_time, "feature lock must strictly precede outcome linkage", AuthorizationError)

    for path, label in (
        (lock_path, "feature lock"),
        (outcomes_path, "outcomes"),
        (phenotypes_path, "phenotypes"),
    ):
        _require(path.is_file(), f"{label} file does not exist: {path}", AuthorizationError)
    expected_hashes = {
        "feature lock": authorization["feature_lock_sha256"],
        "outcomes": authorization["outcomes_sha256"],
        "phenotypes": authorization["phenotypes_sha256"],
    }
    actual_hashes = {
        "feature lock": sha256_file(lock_path),
        "outcomes": sha256_file(outcomes_path),
        "phenotypes": sha256_file(phenotypes_path),
    }
    for label in expected_hashes:
        expected = expected_hashes[label]
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                 f"authorization has invalid {label} SHA-256", AuthorizationError)
        _require(expected == actual_hashes[label], f"{label} SHA-256 does not match authorization", AuthorizationError)


def _expected_structural_phenotype(phenotype_definition: Mapping[str, object]) -> dict[str, object]:
    return {
        "name": phenotype_definition["symbol"],
        **{
            field: phenotype_definition[field]
            for field in (
                "estimator",
                "capacity_per_rotation",
                "within_pv_rotation_aggregation",
                "across_pv_aggregation",
                "operation_order",
                "definition",
                "conductivity_map",
                "full_reference_score",
                "reconstruction_score",
                "probability_clipping_only",
                "log_offset",
                "strict_positive_capacity_required",
                "higher_means",
            )
        },
    }


def validate_provenance_chain(
    lock: Mapping[str, object],
    structural_analysis_lock: Path,
    phenotype_provenance: Path,
) -> None:
    """Verify the two upstream outcome-free artifacts named by the feature lock."""
    for path, field, label in (
        (structural_analysis_lock, "structural_analysis_lock_sha256", "structural analysis lock"),
        (phenotype_provenance, "phenotype_provenance_sha256", "phenotype provenance"),
    ):
        _require(path.is_file(), f"{label} does not exist: {path}")
        actual = sha256_file(path)
        _require(actual == lock[field], f"{label} SHA-256 does not match the recurrence feature lock")

    structural = _read_json(structural_analysis_lock, "structural analysis lock")
    _require_keys_subset(structural, {"schema_version", "status", "frozen_phenotype"}, "structural analysis lock")
    _require(structural["schema_version"] == "1.0", "structural analysis lock schema_version must be 1.0")
    _require(structural["status"] == "LOCKED", "structural analysis lock status must be LOCKED")
    _require(not _contains_replace_marker(structural), "structural analysis lock still contains a REPLACE marker")
    phenotype_definition = lock["phenotype_definition"]
    expected_structural_phenotype = _expected_structural_phenotype(phenotype_definition)
    _require(structural["frozen_phenotype"] == expected_structural_phenotype,
             "structural analysis lock and recurrence feature lock define different phenotypes")

    provenance = _read_json(phenotype_provenance, "phenotype provenance")
    _require_keys(
        provenance,
        {
            "schema_version",
            "scope",
            "n_patients",
            "feature",
            "definition",
            "analysis_lock_sha256",
            "phenotype_table_sha256",
            "long_form_phenotype_sha256",
            "run_provenance_sha256",
            "outcomes_accessed",
        },
        "phenotype provenance",
    )
    _require(provenance["schema_version"] == "1.0", "phenotype provenance schema_version must be 1.0")
    _require(provenance["scope"] == "complete_locked_cohort", "phenotype provenance must cover the complete locked cohort")
    _require(_is_json_integer(provenance["n_patients"])
             and provenance["n_patients"] == lock["expected_patient_count"],
             "phenotype provenance patient count differs from the locked cohort")
    _require(provenance["feature"] == lock["phenotype_definition"]["primary_feature"],
             "phenotype provenance names a different primary feature")
    _require(provenance["definition"] == lock["phenotype_definition"]["definition"],
             "phenotype provenance has a different feature definition")
    _require(provenance["analysis_lock_sha256"] == lock["structural_analysis_lock_sha256"],
             "phenotype provenance does not point to the same structural analysis lock")
    _require(provenance["phenotype_table_sha256"] == lock["phenotype_table_sha256"],
             "phenotype provenance does not bind the locked phenotype table")
    for field in ("long_form_phenotype_sha256", "run_provenance_sha256"):
        value = provenance[field]
        _require(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            and value != "0" * 64,
            f"phenotype provenance {field} must be a completed SHA-256 digest",
        )
    _require(provenance["outcomes_accessed"] is False,
             "phenotype provenance must attest outcomes_accessed=false")


def _read_csv_with_id(path: Path, label: str, string_columns: Sequence[str] = ()) -> pd.DataFrame:
    try:
        dtypes = {"patient_id": "string", **{column: "string" for column in string_columns}}
        frame = pd.read_csv(path, dtype=dtypes, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise DataValidationError(f"cannot read {label}: {error}") from error
    _require("patient_id" in frame.columns, f"{label} lacks patient_id")
    _require(len(frame) > 0, f"{label} has no rows")
    ids = frame["patient_id"]
    _require(bool(ids.notna().all()), f"{label} contains a missing patient_id")
    _require(bool((ids.str.len() > 0).all()), f"{label} contains an empty patient_id")
    _require(bool((ids == ids.str.strip()).all()), f"{label} patient_id has leading or trailing whitespace")
    _require(bool(ids.str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}").all()),
             f"{label} patient_id contains an unsupported character or is too long")
    _require(not bool(ids.duplicated().any()), f"{label} contains duplicate patient_id values")
    return frame


def _parse_date_column(frame: pd.DataFrame, column: str, allow_blank: bool) -> pd.Series:
    raw = frame[column].astype("string")
    if allow_blank:
        blank = raw.str.len() == 0
    else:
        blank = pd.Series(False, index=frame.index)
        _require(bool((raw.str.len() > 0).all()), f"{column} contains a missing date")
    parsed = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if bool((~blank).any()):
        try:
            parsed.loc[~blank] = pd.to_datetime(raw.loc[~blank], format="%Y-%m-%d", errors="raise")
        except (ValueError, TypeError) as error:
            raise DataValidationError(f"{column} must use exact YYYY-MM-DD dates") from error
    return parsed


def validate_outcomes(
    frame: pd.DataFrame,
    analysis_mode: str,
    expected_n: int,
    horizon_days: int,
    blanking_days: int,
) -> tuple[pd.DataFrame, bool]:
    """Validate binary labels and, when supplied, exact longitudinal chronology."""
    present_date_columns = DATE_COLUMNS.intersection(frame.columns)
    _require(not present_date_columns or present_date_columns == DATE_COLUMNS,
             "the four date columns must be all present or all absent")
    has_date_headers = bool(present_date_columns)
    expected_columns = BASE_OUTCOME_COLUMNS | (DATE_COLUMNS if has_date_headers else set())
    _require(set(frame.columns) == expected_columns,
             f"outcomes columns must be exactly {sorted(expected_columns)}")
    _require(len(frame) == expected_n,
             f"outcomes contain {len(frame)} rows but feature lock expects {expected_n}")

    raw_labels = frame["recurrence_2y"].astype("string")
    _require(bool(raw_labels.isin(["0", "1"]).all()),
             "recurrence_2y CSV lexemes must be exactly 0 or 1")
    frame = frame.copy()
    frame["recurrence_2y"] = raw_labels.astype(int)
    positives = int(frame["recurrence_2y"].sum())
    _require(positives >= 5 and len(frame) - positives >= 5,
             "binary analysis requires at least five patients in each outcome class")

    if not has_date_headers:
        _require(analysis_mode == "binary", "post-MRI landmark analysis requires all four date columns")
        return frame, False

    all_date_cells_blank = all(
        bool((frame[column].astype("string").str.len() == 0).all()) for column in DATE_COLUMNS
    )
    if all_date_cells_blank:
        _require(analysis_mode == "binary",
                 "post-MRI landmark analysis requires populated dates and censoring information")
        return frame, False

    for column in ("ablation_date", "post_ablation_mri_date", "followup_end_date"):
        frame[column] = _parse_date_column(frame, column, allow_blank=False)
    frame["recurrence_date"] = _parse_date_column(frame, "recurrence_date", allow_blank=True)

    ablation = frame["ablation_date"]
    mri = frame["post_ablation_mri_date"]
    followup = frame["followup_end_date"]
    recurrence = frame["recurrence_date"]
    _require(bool((mri > ablation).all()), "every post-ablation MRI date must be after ablation")
    _require(bool((followup > mri).all()), "every follow-up end date must be after post-ablation MRI")
    has_recurrence = recurrence.notna()
    _require(bool((recurrence.loc[has_recurrence] <= followup.loc[has_recurrence]).all()),
             "a recurrence date occurs after follow-up ended")
    qualifying_start = ablation + pd.to_timedelta(blanking_days, unit="D")
    _require(bool((recurrence.loc[has_recurrence] >= qualifying_start.loc[has_recurrence]).all()),
             "recurrence_date must be the first qualifying event on or after the locked blanking period")

    binary_end = ablation + pd.to_timedelta(horizon_days, unit="D")
    event_by_horizon = has_recurrence & (recurrence <= binary_end)
    label_one = frame["recurrence_2y"] == 1
    _require(bool((label_one == event_by_horizon).all()),
             "recurrence_2y disagrees with recurrence_date and the locked two-year horizon")
    adequate_non_event_followup = (followup >= binary_end) | (has_recurrence & (recurrence > binary_end))
    _require(bool(adequate_non_event_followup.loc[~label_one].all()),
             "a two-year non-event label lacks two years of observed follow-up")
    return frame, True


def _transform_feature(values: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    _require(bool(np.isfinite(values).all()), f"feature {spec.name} contains a missing or non-finite value")
    if spec.transform == "identity":
        transformed = values
    elif spec.transform == "log1p_nonnegative":
        _require(bool((values >= 0.0).all()), f"feature {spec.name} must be nonnegative for log1p")
        transformed = np.log1p(values)
    elif spec.transform == "log_positive":
        _require(bool((values > 0.0).all()), f"feature {spec.name} must be positive for log")
        transformed = np.log(values)
    elif spec.transform == "sqrt_nonnegative":
        _require(bool((values >= 0.0).all()), f"feature {spec.name} must be nonnegative for square root")
        transformed = np.sqrt(values)
    else:  # protected by validate_feature_lock
        raise DataValidationError(f"unsupported transformation {spec.transform}")
    _require(bool(np.isfinite(transformed).all()), f"transformed feature {spec.name} is non-finite")
    return transformed


def validate_and_merge_tables(
    outcomes: pd.DataFrame,
    phenotypes: pd.DataFrame,
    features: Sequence[FeatureSpec],
    expected_n: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    expected_phenotype_columns = {"patient_id", *(feature.name for feature in features)}
    _require(set(phenotypes.columns) == expected_phenotype_columns,
             f"phenotype columns must be exactly {sorted(expected_phenotype_columns)}")
    _require(len(phenotypes) == expected_n,
             f"phenotypes contain {len(phenotypes)} rows but feature lock expects {expected_n}")
    outcome_ids = set(outcomes["patient_id"])
    phenotype_ids = set(phenotypes["patient_id"])
    _require(outcome_ids == phenotype_ids,
             f"ID sets differ: {len(outcome_ids - phenotype_ids)} outcome-only and "
             f"{len(phenotype_ids - outcome_ids)} phenotype-only IDs")
    merged = outcomes.merge(phenotypes, on="patient_id", how="inner", validate="one_to_one")
    merged = merged.sort_values("patient_id", kind="mergesort").reset_index(drop=True)
    transformed: dict[str, np.ndarray] = {}
    for feature in features:
        numeric = pd.to_numeric(merged[feature.name], errors="coerce").to_numpy(dtype=float)
        transformed[feature.name] = _transform_feature(numeric, feature)
    return merged, transformed


def partition_indices(
    merged: pd.DataFrame,
    partition: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the locked partition against the linked table without reordering it."""
    development_ids = set(partition["development_ids"])
    holdout_ids = set(partition["holdout_ids"])
    observed_ids = set(merged["patient_id"])
    _require(
        observed_ids == development_ids | holdout_ids,
        "linked patient IDs do not equal the locked development/holdout partition",
    )
    development = merged["patient_id"].isin(development_ids).to_numpy(dtype=bool)
    holdout = merged["patient_id"].isin(holdout_ids).to_numpy(dtype=bool)
    _require(not bool(np.any(development & holdout)), "a linked patient belongs to both partitions")
    _require(bool(np.all(development | holdout)), "a linked patient belongs to neither partition")
    _require(int(np.sum(development)) == len(development_ids), "development partition count changed after linkage")
    _require(int(np.sum(holdout)) == len(holdout_ids), "holdout partition count changed after linkage")
    return development, holdout


def _standardize_train_test(train: np.ndarray, test: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if train.shape[1] == 0:
        return train.copy(), test.copy(), np.empty(0), np.empty(0)
    means = np.mean(train, axis=0)
    scales = np.std(train, axis=0, ddof=1)
    for name, scale in zip(names, scales, strict=True):
        if not np.isfinite(scale) or scale <= 1.0e-12:
            raise ModelFitError(f"feature {name} is constant or numerically degenerate in a training fold")
    return (train - means) / scales, (test - means) / scales, means, scales


def _firth_components(design: np.ndarray, outcome: np.ndarray, beta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    eta = design @ beta
    probability = expit(eta)
    weight = np.maximum(probability * (1.0 - probability), 1.0e-12)
    information = design.T @ (weight[:, None] * design)
    sign, logdet = np.linalg.slogdet(information)
    if sign <= 0 or not np.isfinite(logdet):
        raise ModelFitError("Firth information matrix is singular")
    inverse = np.linalg.inv(information)
    leverage = weight * np.einsum("ij,jk,ik->i", design, inverse, design)
    log_likelihood = float(np.sum(outcome * eta - np.logaddexp(0.0, eta)))
    penalized = log_likelihood + 0.5 * float(logdet)
    adjusted_score = design.T @ (outcome - probability + leverage * (0.5 - probability))
    return penalized, adjusted_score, information


def _require_stationary(gradient: np.ndarray, label: str) -> None:
    _require(bool(np.isfinite(gradient).all()), f"{label} stationarity vector is non-finite", ModelFitError)
    maximum = float(np.max(np.abs(gradient))) if gradient.size else 0.0
    _require(maximum <= STATIONARITY_TOLERANCE,
             f"{label} failed recomputed stationarity: max absolute score {maximum:.3e} "
             f"> {STATIONARITY_TOLERANCE:.3e}", ModelFitError)


def fit_firth_logistic(features: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit Jeffreys-penalised logistic regression, including an intercept."""
    design = np.column_stack([np.ones(len(outcome)), features])
    _require(np.linalg.matrix_rank(design) == design.shape[1],
             "binary design matrix is rank deficient", ModelFitError)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        penalized, score, _ = _firth_components(design, outcome, beta)
        return -penalized, -score

    result = minimize(
        objective,
        np.zeros(design.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1.0e-12, "gtol": 1.0e-8, "maxls": 50},
    )
    beta = np.asarray(result.x, dtype=float)
    _, score, information = _firth_components(design, outcome, beta)
    _require(bool(np.isfinite(beta).all()), "Firth logistic coefficients are non-finite", ModelFitError)
    try:
        _require_stationary(score, "Firth logistic fit")
    except ModelFitError as error:
        raise ModelFitError(f"{error}; optimizer status: {result.message}") from error
    covariance = np.linalg.inv(information)
    return beta, covariance


def _model_feature_sets(features: Sequence[FeatureSpec]) -> dict[str, list[str]]:
    phenotype = [feature.name for feature in features if feature.role == "phenotype"]
    clinical = [feature.name for feature in features if feature.role == "clinical"]
    models: dict[str, list[str]] = {"null": [], "phenotype": phenotype}
    if clinical:
        models["clinical"] = clinical
        models["clinical_plus_phenotype"] = clinical + phenotype
    return models


def _matrix_for_names(transformed: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    if not names:
        n = len(next(iter(transformed.values())))
        return np.empty((n, 0), dtype=float)
    return np.column_stack([transformed[name] for name in names])


def leave_one_out_binary_predictions(
    matrix: np.ndarray,
    outcome: np.ndarray,
    names: Sequence[str],
) -> np.ndarray:
    prediction = np.empty(len(outcome), dtype=float)
    for held_out in range(len(outcome)):
        train_mask = np.ones(len(outcome), dtype=bool)
        train_mask[held_out] = False
        train_y = outcome[train_mask]
        if min(int(train_y.sum()), int(len(train_y) - train_y.sum())) < 4:
            raise ModelFitError("a leave-one-patient-out binary training fold has fewer than four patients in one class")
        train_x, test_x, _, _ = _standardize_train_test(
            matrix[train_mask], matrix[[held_out]], names
        )
        beta, _ = fit_firth_logistic(train_x, train_y)
        prediction[held_out] = float(expit(np.r_[1.0, test_x[0]] @ beta))
    _require(bool(np.isfinite(prediction).all()), "cross-fitted binary prediction is non-finite", ModelFitError)
    return np.clip(prediction, 1.0e-10, 1.0 - 1.0e-10)


def _binary_auc(outcome: np.ndarray, probability: np.ndarray) -> float:
    positives = outcome == 1
    n_positive = int(np.sum(positives))
    n_negative = len(outcome) - n_positive
    if n_positive == 0 or n_negative == 0:
        return math.nan
    ranks = rankdata(probability, method="average")
    return float((np.sum(ranks[positives]) - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def _average_precision(outcome: np.ndarray, probability: np.ndarray) -> float:
    n_positive = int(np.sum(outcome))
    if n_positive == 0:
        return math.nan
    order = np.argsort(-probability, kind="mergesort")
    ordered = outcome[order]
    ordered_probability = probability[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(ordered_probability) != 0.0), len(outcome) - 1]
    true_positives = np.cumsum(ordered)[threshold_ends]
    precision = true_positives / (threshold_ends + 1)
    recall = true_positives / n_positive
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def _binary_log_loss(outcome: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1.0e-10, 1.0 - 1.0e-10)
    return float(-np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log1p(-clipped)))


def binary_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1.0e-10, 1.0 - 1.0e-10)
    prevalence = float(np.mean(outcome))
    if prevalence <= 0.0 or prevalence >= 1.0:
        calibration_in_the_large = math.nan
    else:
        offset = np.log(clipped) - np.log1p(-clipped)
        calibration_in_the_large = float(brentq(
            lambda intercept: float(np.mean(expit(offset + intercept)) - prevalence),
            -50.0,
            50.0,
        ))
    return {
        "roc_auc": _binary_auc(outcome, clipped),
        "average_precision": _average_precision(outcome, clipped),
        "brier_score": float(np.mean((outcome - clipped) ** 2)),
        "log_loss": _binary_log_loss(outcome, clipped),
        "calibration_in_the_large": calibration_in_the_large,
    }


def exact_holdout_log_loss_randomization(
    outcome: np.ndarray,
    phenotype_probability: np.ndarray,
    null_probability: np.ndarray,
) -> pd.DataFrame:
    """Enumerate fixed-count holdout labels for the frozen M1-minus-M0 score."""
    n_patients = len(outcome)
    n_events = int(np.sum(outcome))
    number_assignments = math.comb(n_patients, n_events)
    _require(number_assignments <= 200000,
             "exact holdout randomization exceeds the fixed enumeration safety limit", ModelFitError)
    observed = _binary_log_loss(outcome, phenotype_probability) - _binary_log_loss(outcome, null_probability)
    null_statistics = np.empty(number_assignments, dtype=float)
    label = np.zeros(n_patients, dtype=int)
    for index, event_positions in enumerate(itertools.combinations(range(n_patients), n_events)):
        label.fill(0)
        label[list(event_positions)] = 1
        null_statistics[index] = (
            _binary_log_loss(label, phenotype_probability) - _binary_log_loss(label, null_probability)
        )
    _require(float(np.min(np.abs(null_statistics - observed))) <= 1.0e-12,
             "observed holdout allocation is absent from exact enumeration", ModelFitError)
    p_value = float(np.mean(null_statistics <= observed + 1.0e-15))
    return pd.DataFrame([{
        "observed_delta_log_loss_m1_minus_m0": observed,
        "exact_null_q025": float(np.quantile(null_statistics, 0.025)),
        "exact_null_median": float(np.quantile(null_statistics, 0.5)),
        "exact_null_q975": float(np.quantile(null_statistics, 0.975)),
        "exact_one_sided_p_negative_delta": p_value,
        "n_holdout_patients": n_patients,
        "n_holdout_events": n_events,
        "number_fixed_count_assignments": number_assignments,
        "randomization_unit": "holdout_patient_outcome_label",
        "prediction_handling": "development_fitted_m0_m1_predictions_held_fixed",
        "favorable_direction": "negative",
    }])


def _percentile_interval(values: Sequence[float]) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    _require(len(finite) > 0, "no finite bootstrap replicates", ModelFitError)
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(len(finite))


def _bootstrap_indices(n: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(replicates, n), endpoint=False)


def fit_binary_models(
    merged: pd.DataFrame,
    transformed: Mapping[str, np.ndarray],
    features: Sequence[FeatureSpec],
    development_mask: np.ndarray,
    holdout_mask: np.ndarray,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    permutation_replicates: int,
    permutation_seed: int,
) -> dict[str, pd.DataFrame]:
    outcome_all = merged["recurrence_2y"].to_numpy(dtype=int)
    development_outcome = outcome_all[development_mask]
    holdout_outcome = outcome_all[holdout_mask]
    model_features = _model_feature_sets(features)
    holdout_predictions: dict[str, np.ndarray] = {}
    development_predictions: dict[str, np.ndarray] = {}
    development_matrices: dict[str, np.ndarray] = {}
    coefficient_rows: list[dict[str, object]] = []

    for model, names in model_features.items():
        matrix_all = _matrix_for_names(transformed, names)
        development_x = matrix_all[development_mask]
        development_matrices[model] = development_x
        holdout_x = matrix_all[holdout_mask]
        scaled_development, scaled_holdout, means, scales = _standardize_train_test(
            development_x, holdout_x, names
        )
        interquartile_ranges = (
            np.quantile(development_x, 0.75, axis=0) - np.quantile(development_x, 0.25, axis=0)
            if names else np.empty(0)
        )
        beta, covariance = fit_firth_logistic(scaled_development, development_outcome)
        holdout_predictions[model] = np.clip(
            expit(np.column_stack([np.ones(len(scaled_holdout)), scaled_holdout]) @ beta),
            1.0e-10,
            1.0 - 1.0e-10,
        )
        development_predictions[model] = leave_one_out_binary_predictions(
            development_x, development_outcome, names
        )
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        coefficient_names = ["intercept", *names]
        for index, coefficient_name in enumerate(coefficient_names):
            mean = math.nan if index == 0 else float(means[index - 1])
            scale = math.nan if index == 0 else float(scales[index - 1])
            coefficient = float(beta[index])
            standard_error = float(standard_errors[index])
            if index == 0:
                interquartile_range = math.nan
                odds_ratio_iqr = math.nan
                odds_ratio_iqr_low = math.nan
                odds_ratio_iqr_high = math.nan
            else:
                interquartile_range = float(interquartile_ranges[index - 1])
                iqr_in_sd = interquartile_range / float(scales[index - 1])
                odds_ratio_iqr = float(np.exp(coefficient * iqr_in_sd))
                odds_ratio_iqr_low = float(np.exp((coefficient - 1.96 * standard_error) * iqr_in_sd))
                odds_ratio_iqr_high = float(np.exp((coefficient + 1.96 * standard_error) * iqr_in_sd))
            coefficient_rows.append({
                "model": model,
                "term": coefficient_name,
                "coefficient": coefficient,
                "odds_ratio": float(np.exp(coefficient)),
                "approximate_information_standard_error": standard_error,
                "approximate_wald_ci_low": float(np.exp(coefficient - 1.96 * standard_error)),
                "approximate_wald_ci_high": float(np.exp(coefficient + 1.96 * standard_error)),
                "transformed_training_mean": mean,
                "transformed_training_sd": scale,
                "transformed_development_iqr": interquartile_range,
                "odds_ratio_per_development_iqr": odds_ratio_iqr,
                "odds_ratio_per_development_iqr_ci_low": odds_ratio_iqr_low,
                "odds_ratio_per_development_iqr_ci_high": odds_ratio_iqr_high,
                "estimator": LOCKED_BINARY_ESTIMATOR,
                "fit_partition": "development",
            })

    comparison_pairs: list[tuple[str, str]] = [("phenotype", "null")]
    primary_pair = ("phenotype", "null")
    if "clinical" in holdout_predictions:
        comparison_pairs.extend([
            ("clinical", "null"),
            ("clinical_plus_phenotype", "clinical"),
        ])

    def metric_table(
        outcome: np.ndarray,
        predictions: Mapping[str, np.ndarray],
        bootstrap_indices: np.ndarray,
        evaluation: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for model, prediction in predictions.items():
            estimates = binary_metrics(outcome, prediction)
            bootstrap_values = {metric: [] for metric in estimates}
            for sample in bootstrap_indices:
                sampled = binary_metrics(outcome[sample], prediction[sample])
                for metric, value in sampled.items():
                    bootstrap_values[metric].append(value)
            for metric, estimate in estimates.items():
                low, high, valid = _percentile_interval(bootstrap_values[metric])
                _require(valid >= int(0.9 * len(bootstrap_indices)),
                         f"too few valid patient bootstrap replicates for {evaluation} {model} {metric}", ModelFitError)
                rows.append({
                    "model": model,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_patients": len(outcome),
                    "n_recurrence_2y": int(outcome.sum()),
                    "evaluation": evaluation,
                    "interval": "patient_percentile_bootstrap_on_fixed_predictions",
                    "valid_bootstrap_replicates": valid,
                })
        return pd.DataFrame(rows)

    def comparison_table(
        outcome: np.ndarray,
        predictions: Mapping[str, np.ndarray],
        bootstrap_indices: np.ndarray,
        evaluation: str,
    ) -> pd.DataFrame:
        comparison_rows: list[dict[str, object]] = []
        definitions: dict[str, Callable[[dict[str, float], dict[str, float]], float]] = {
            "delta_roc_auc": lambda new, old: new["roc_auc"] - old["roc_auc"],
            "delta_average_precision": lambda new, old: new["average_precision"] - old["average_precision"],
            "brier_improvement": lambda new, old: old["brier_score"] - new["brier_score"],
            "log_loss_improvement": lambda new, old: old["log_loss"] - new["log_loss"],
            "delta_log_loss_candidate_minus_reference": lambda new, old: new["log_loss"] - old["log_loss"],
        }
        for candidate, reference in comparison_pairs:
            candidate_estimate = binary_metrics(outcome, predictions[candidate])
            reference_estimate = binary_metrics(outcome, predictions[reference])
            for metric, operation in definitions.items():
                estimate = operation(candidate_estimate, reference_estimate)
                values: list[float] = []
                for sample in bootstrap_indices:
                    new = binary_metrics(outcome[sample], predictions[candidate][sample])
                    old = binary_metrics(outcome[sample], predictions[reference][sample])
                    values.append(operation(new, old))
                low, high, valid = _percentile_interval(values)
                _require(valid >= int(0.9 * len(bootstrap_indices)),
                         f"too few valid paired comparisons for {evaluation} {candidate} vs {reference}", ModelFitError)
                comparison_rows.append({
                    "candidate_model": candidate,
                    "reference_model": reference,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_patients": len(outcome),
                    "n_recurrence_2y": int(outcome.sum()),
                    "evaluation": evaluation,
                    "is_primary_incremental_comparison": int((candidate, reference) == primary_pair),
                    "is_primary_endpoint": int(
                        (candidate, reference) == primary_pair
                        and metric == "delta_log_loss_candidate_minus_reference"
                        and evaluation == LOCKED_PRIMARY_EVALUATION
                    ),
                    "favorable_direction": (
                        "negative" if metric == "delta_log_loss_candidate_minus_reference" else "positive"
                    ),
                    "interval": "paired_patient_percentile_bootstrap",
                    "valid_bootstrap_replicates": valid,
                })
        return pd.DataFrame(comparison_rows)

    holdout_bootstrap = _bootstrap_indices(len(holdout_outcome), bootstrap_replicates, bootstrap_seed)
    development_bootstrap = _bootstrap_indices(
        len(development_outcome), bootstrap_replicates, bootstrap_seed + 1
    )
    holdout_prediction_frame = pd.DataFrame({
        "patient_id": merged.loc[holdout_mask, "patient_id"].to_numpy(),
        "recurrence_2y": holdout_outcome,
        "partition": "holdout",
        **{f"probability_{model}": value for model, value in holdout_predictions.items()},
    })
    development_prediction_frame = pd.DataFrame({
        "patient_id": merged.loc[development_mask, "patient_id"].to_numpy(),
        "recurrence_2y": development_outcome,
        "partition": "development",
        **{f"probability_{model}": value for model, value in development_predictions.items()},
    })
    exact_holdout_randomization = exact_holdout_log_loss_randomization(
        holdout_outcome,
        holdout_predictions["phenotype"],
        holdout_predictions["null"],
    )
    observed_primary_delta = (
        _binary_log_loss(development_outcome, development_predictions["phenotype"])
        - _binary_log_loss(development_outcome, development_predictions["null"])
    )
    permutation_rng = np.random.default_rng(permutation_seed)
    permutation_deltas = np.empty(permutation_replicates, dtype=float)
    for replicate in range(permutation_replicates):
        permuted_outcome = permutation_rng.permutation(development_outcome)
        null_probability = leave_one_out_binary_predictions(
            development_matrices["null"], permuted_outcome, []
        )
        phenotype_names = model_features["phenotype"]
        phenotype_probability = leave_one_out_binary_predictions(
            development_matrices["phenotype"], permuted_outcome, phenotype_names
        )
        permutation_deltas[replicate] = (
            _binary_log_loss(permuted_outcome, phenotype_probability)
            - _binary_log_loss(permuted_outcome, null_probability)
        )
    permutation_p = (1.0 + float(np.sum(permutation_deltas <= observed_primary_delta))) / (
        permutation_replicates + 1.0
    )
    permutation_draws = pd.DataFrame({
        "replicate": np.arange(permutation_replicates, dtype=int),
        "permuted_delta_log_loss_m1_minus_m0": permutation_deltas,
        "permutation_seed": permutation_seed,
    })
    permutation_summary = pd.DataFrame([{
        "observed_development_loo_delta_log_loss_m1_minus_m0": observed_primary_delta,
        "permutation_null_q025": float(np.quantile(permutation_deltas, 0.025)),
        "permutation_null_median": float(np.quantile(permutation_deltas, 0.5)),
        "permutation_null_q975": float(np.quantile(permutation_deltas, 0.975)),
        "one_sided_p_negative_delta": permutation_p,
        "permutation_replicates": permutation_replicates,
        "permutation_seed": permutation_seed,
        "permutation_unit": "development_patient_outcome_label",
        "evaluation": LOCKED_CROSS_FITTING,
    }])
    return {
        "binary_holdout_patient_predictions.csv": holdout_prediction_frame,
        "binary_holdout_metrics.csv": metric_table(
            holdout_outcome, holdout_predictions, holdout_bootstrap, LOCKED_PRIMARY_EVALUATION
        ),
        "binary_holdout_model_comparisons.csv": comparison_table(
            holdout_outcome, holdout_predictions, holdout_bootstrap, LOCKED_PRIMARY_EVALUATION
        ),
        "binary_holdout_primary_exact_randomization.csv": exact_holdout_randomization,
        "binary_development_loo_predictions.csv": development_prediction_frame,
        "binary_development_loo_metrics.csv": metric_table(
            development_outcome, development_predictions, development_bootstrap, LOCKED_CROSS_FITTING
        ),
        "binary_development_loo_model_comparisons.csv": comparison_table(
            development_outcome, development_predictions, development_bootstrap, LOCKED_CROSS_FITTING
        ),
        "binary_development_coefficients.csv": pd.DataFrame(coefficient_rows),
        "binary_development_primary_permutation_draws.csv": permutation_draws,
        "binary_development_primary_permutation_summary.csv": permutation_summary,
    }


def make_landmark_cohort(merged: pd.DataFrame, horizon_days: int = 730) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = merged.copy()
    recurrence = working["recurrence_date"]
    mri = working["post_ablation_mri_date"]
    followup = working["followup_end_date"]
    administrative_end = working["ablation_date"] + pd.to_timedelta(horizon_days, unit="D")
    analysis_end = pd.concat([followup, administrative_end], axis=1).min(axis=1)
    pre_landmark_event = recurrence.notna() & (recurrence <= mri)
    no_post_landmark_window = analysis_end <= mri
    eligible = ~pre_landmark_event & ~no_post_landmark_window
    reasons = np.select(
        [pre_landmark_event, no_post_landmark_window],
        ["excluded_qualifying_recurrence_on_or_before_mri", "excluded_no_followup_before_two_year_administrative_end"],
        default="eligible_event_free_at_post_ablation_mri",
    )
    eligibility = pd.DataFrame({
        "patient_id": working["patient_id"],
        "landmark_eligible": eligible.astype(int),
        "eligibility_reason": reasons,
    })
    working["landmark_analysis_end"] = analysis_end
    landmark = working.loc[eligible].copy().reset_index(drop=True)
    post_landmark_event = (
        landmark["recurrence_date"].notna()
        & (landmark["recurrence_date"] > landmark["post_ablation_mri_date"])
        & (landmark["recurrence_date"] <= landmark["landmark_analysis_end"])
    )
    event = post_landmark_event.to_numpy(dtype=int)
    endpoint = landmark["recurrence_date"].where(post_landmark_event, landmark["landmark_analysis_end"])
    time_days = (endpoint - landmark["post_ablation_mri_date"]).dt.days.to_numpy(dtype=float)
    _require(bool((time_days > 0.0).all()), "landmark follow-up time must be strictly positive")
    landmark["landmark_time_days"] = time_days
    landmark["landmark_event"] = event
    _require(len(landmark) >= 20, "landmark analysis requires at least 20 eligible patients")
    _require(int(event.sum()) >= 5 and len(event) - int(event.sum()) >= 5,
             "landmark analysis requires at least five events and five censored observations")
    return landmark, eligibility


def _cox_components(matrix: np.ndarray, time: np.ndarray, event: np.ndarray, beta: np.ndarray, ridge: float) -> tuple[float, np.ndarray, np.ndarray]:
    linear = matrix @ beta
    log_likelihood = 0.0
    gradient = np.zeros(matrix.shape[1])
    information = np.zeros((matrix.shape[1], matrix.shape[1]))
    event_times = np.unique(time[event == 1])
    for event_time in event_times:
        event_mask = (time == event_time) & (event == 1)
        risk_mask = time >= event_time
        number_events = int(np.sum(event_mask))
        risk_linear = linear[risk_mask]
        log_denominator = float(logsumexp(risk_linear))
        weights = np.exp(risk_linear - log_denominator)
        risk_x = matrix[risk_mask]
        weighted_mean = np.sum(weights[:, None] * risk_x, axis=0)
        weighted_second = np.einsum("i,ij,ik->jk", weights, risk_x, risk_x)
        covariance = weighted_second - np.outer(weighted_mean, weighted_mean)
        log_likelihood += float(np.sum(linear[event_mask])) - number_events * log_denominator
        gradient += np.sum(matrix[event_mask], axis=0) - number_events * weighted_mean
        information += number_events * covariance
    log_likelihood -= 0.5 * ridge * float(beta @ beta)
    gradient -= ridge * beta
    information += ridge * np.eye(matrix.shape[1])
    return log_likelihood, gradient, information


def fit_ridge_cox(matrix: np.ndarray, time: np.ndarray, event: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    _require(matrix.shape[1] > 0, "Cox model requires at least one feature", ModelFitError)
    _require(np.linalg.matrix_rank(matrix) == matrix.shape[1], "Cox design matrix is rank deficient", ModelFitError)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        likelihood, gradient, _ = _cox_components(matrix, time, event, beta, ridge)
        return -likelihood, -gradient

    result = minimize(
        objective,
        np.zeros(matrix.shape[1]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1.0e-12, "gtol": 1.0e-8, "maxls": 50},
    )
    beta = np.asarray(result.x, dtype=float)
    _, gradient, information = _cox_components(matrix, time, event, beta, ridge)
    _require(bool(np.isfinite(beta).all()), "ridge Cox coefficients are non-finite", ModelFitError)
    try:
        _require_stationary(gradient, "ridge Cox fit")
    except ModelFitError as error:
        raise ModelFitError(f"{error}; optimizer status: {result.message}") from error
    covariance = np.linalg.inv(information)
    return beta, covariance


def leave_one_out_cox_scores(
    matrix: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    names: Sequence[str],
    ridge: float,
) -> np.ndarray:
    score = np.empty(len(event), dtype=float)
    for held_out in range(len(event)):
        train_mask = np.ones(len(event), dtype=bool)
        train_mask[held_out] = False
        train_event = event[train_mask]
        if int(train_event.sum()) < 4:
            raise ModelFitError("a landmark training fold contains fewer than four events")
        train_x, test_x, _, _ = _standardize_train_test(matrix[train_mask], matrix[[held_out]], names)
        beta, _ = fit_ridge_cox(train_x, time[train_mask], train_event, ridge)
        score[held_out] = float(test_x[0] @ beta)
    _require(bool(np.isfinite(score).all()), "cross-fitted Cox score is non-finite", ModelFitError)
    return score - float(np.mean(score))


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0
    for left in range(len(time) - 1):
        for right in range(left + 1, len(time)):
            if time[left] < time[right] and event[left] == 1:
                earlier, later = left, right
            elif time[right] < time[left] and event[right] == 1:
                earlier, later = right, left
            else:
                continue
            comparable += 1
            if risk[earlier] > risk[later]:
                concordant += 1.0
            elif risk[earlier] == risk[later]:
                concordant += 0.5
    return math.nan if comparable == 0 else float(concordant / comparable)


def fit_landmark_models(
    merged: pd.DataFrame,
    transformed_all: Mapping[str, np.ndarray],
    features: Sequence[FeatureSpec],
    development_mask_all: np.ndarray,
    holdout_mask_all: np.ndarray,
    ridge: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, pd.DataFrame]:
    landmark, eligibility = make_landmark_cohort(merged)
    eligibility["partition"] = np.where(development_mask_all, "development", "holdout")
    eligible_ids = set(landmark["patient_id"])
    eligible_indices = np.array([patient_id in eligible_ids for patient_id in merged["patient_id"]], dtype=bool)
    transformed = {name: values[eligible_indices] for name, values in transformed_all.items()}
    development_ids = set(merged.loc[development_mask_all, "patient_id"])
    holdout_ids = set(merged.loc[holdout_mask_all, "patient_id"])
    development_mask = landmark["patient_id"].isin(development_ids).to_numpy(dtype=bool)
    holdout_mask = landmark["patient_id"].isin(holdout_ids).to_numpy(dtype=bool)
    _require(bool(np.all(development_mask | holdout_mask)), "an eligible landmark patient lacks a locked partition")
    time_all = landmark["landmark_time_days"].to_numpy(dtype=float)
    event_all = landmark["landmark_event"].to_numpy(dtype=int)
    development_time = time_all[development_mask]
    development_event = event_all[development_mask]
    holdout_time = time_all[holdout_mask]
    holdout_event = event_all[holdout_mask]
    minimum_development_events = max(5, 3 * len(features))
    _require(len(development_event) >= 20, "landmark development set has fewer than 20 eligible patients")
    _require(int(development_event.sum()) >= minimum_development_events,
             f"landmark development set requires at least {minimum_development_events} events")
    _require(len(development_event) - int(development_event.sum()) >= 5,
             "landmark development set requires at least five censored patients")
    _require(len(holdout_event) >= 8, "landmark holdout has fewer than eight eligible patients")
    _require(int(holdout_event.sum()) >= 2 and len(holdout_event) - int(holdout_event.sum()) >= 2,
             "landmark holdout requires at least two events and two censored patients")

    model_features = _model_feature_sets(features)
    model_features.pop("null")
    holdout_scores: dict[str, np.ndarray] = {"null": np.zeros(len(holdout_event))}
    development_scores: dict[str, np.ndarray] = {"null": np.zeros(len(development_event))}
    coefficient_rows: list[dict[str, object]] = []

    for model, names in model_features.items():
        matrix_all = _matrix_for_names(transformed, names)
        development_x = matrix_all[development_mask]
        holdout_x = matrix_all[holdout_mask]
        scaled_development, scaled_holdout, means, scales = _standardize_train_test(
            development_x, holdout_x, names
        )
        beta, covariance = fit_ridge_cox(
            scaled_development, development_time, development_event, ridge
        )
        holdout_scores[model] = scaled_holdout @ beta
        development_scores[model] = leave_one_out_cox_scores(
            development_x, development_time, development_event, names, ridge
        )
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        for index, name in enumerate(names):
            coefficient = float(beta[index])
            standard_error = float(standard_errors[index])
            coefficient_rows.append({
                "model": model,
                "term": name,
                "coefficient_per_sd": coefficient,
                "hazard_ratio_per_sd": float(np.exp(coefficient)),
                "approximate_penalized_information_standard_error": standard_error,
                "approximate_penalized_wald_ci_low": float(np.exp(coefficient - 1.96 * standard_error)),
                "approximate_penalized_wald_ci_high": float(np.exp(coefficient + 1.96 * standard_error)),
                "coefficient_interval_caveat": "approximate inverse-penalized-information interval",
                "transformed_training_mean": float(means[index]),
                "transformed_training_sd": float(scales[index]),
                "estimator": LOCKED_SURVIVAL_ESTIMATOR,
                "cox_ridge": ridge,
                "fit_partition": "development",
            })

    comparison_pairs: list[tuple[str, str]] = [("phenotype", "null")]
    primary_pair = ("phenotype", "null")
    if "clinical" in holdout_scores:
        comparison_pairs.extend([
            ("clinical", "null"),
            ("clinical_plus_phenotype", "clinical"),
        ])

    def metric_table(
        time: np.ndarray,
        event: np.ndarray,
        scores: Mapping[str, np.ndarray],
        bootstrap: np.ndarray,
        evaluation: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for model, score in scores.items():
            estimate = harrell_c_index(time, event, score)
            values = [harrell_c_index(time[sample], event[sample], score[sample]) for sample in bootstrap]
            low, high, valid = _percentile_interval(values)
            _require(valid >= int(0.9 * len(bootstrap)),
                     f"too few valid landmark replicates for {evaluation} {model}", ModelFitError)
            rows.append({
                "model": model,
                "metric": "harrell_c_index",
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "n_landmark_patients": len(event),
                "n_post_landmark_events": int(event.sum()),
                "evaluation": evaluation if model != "null" else f"{evaluation}_constant_reference",
                "interval": "patient_percentile_bootstrap_on_fixed_scores",
                "valid_bootstrap_replicates": valid,
            })
        return pd.DataFrame(rows)

    def comparison_table(
        time: np.ndarray,
        event: np.ndarray,
        scores: Mapping[str, np.ndarray],
        bootstrap: np.ndarray,
        evaluation: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for candidate, reference in comparison_pairs:
            estimate = harrell_c_index(time, event, scores[candidate]) - harrell_c_index(time, event, scores[reference])
            values = [
                harrell_c_index(time[sample], event[sample], scores[candidate][sample])
                - harrell_c_index(time[sample], event[sample], scores[reference][sample])
                for sample in bootstrap
            ]
            low, high, valid = _percentile_interval(values)
            _require(valid >= int(0.9 * len(bootstrap)),
                     f"too few valid landmark comparisons for {evaluation} {candidate} vs {reference}", ModelFitError)
            rows.append({
                "candidate_model": candidate,
                "reference_model": reference,
                "metric": "delta_harrell_c_index",
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "n_landmark_patients": len(event),
                "n_post_landmark_events": int(event.sum()),
                "evaluation": evaluation,
                "is_primary_incremental_comparison": int((candidate, reference) == primary_pair),
                "interval": "paired_patient_percentile_bootstrap",
                "valid_bootstrap_replicates": valid,
            })
        return pd.DataFrame(rows)

    holdout_bootstrap = _bootstrap_indices(len(holdout_event), bootstrap_replicates, bootstrap_seed + 2)
    development_bootstrap = _bootstrap_indices(
        len(development_event), bootstrap_replicates, bootstrap_seed + 3
    )
    holdout_score_frame = pd.DataFrame({
        "patient_id": landmark.loc[holdout_mask, "patient_id"].to_numpy(),
        "landmark_time_days": holdout_time,
        "landmark_event": holdout_event,
        "partition": "holdout",
        **{f"risk_score_{model}": value for model, value in holdout_scores.items()},
    })
    development_score_frame = pd.DataFrame({
        "patient_id": landmark.loc[development_mask, "patient_id"].to_numpy(),
        "landmark_time_days": development_time,
        "landmark_event": development_event,
        "partition": "development",
        **{f"risk_score_{model}": value for model, value in development_scores.items()},
    })
    return {
        "landmark_eligibility.csv": eligibility,
        "landmark_holdout_patient_scores.csv": holdout_score_frame,
        "landmark_holdout_metrics.csv": metric_table(
            holdout_time, holdout_event, holdout_scores, holdout_bootstrap, LOCKED_PRIMARY_EVALUATION
        ),
        "landmark_holdout_model_comparisons.csv": comparison_table(
            holdout_time, holdout_event, holdout_scores, holdout_bootstrap, LOCKED_PRIMARY_EVALUATION
        ),
        "landmark_development_loo_scores.csv": development_score_frame,
        "landmark_development_loo_metrics.csv": metric_table(
            development_time, development_event, development_scores, development_bootstrap, LOCKED_CROSS_FITTING
        ),
        "landmark_development_loo_model_comparisons.csv": comparison_table(
            development_time, development_event, development_scores, development_bootstrap, LOCKED_CROSS_FITTING
        ),
        "landmark_development_coefficients.csv": pd.DataFrame(coefficient_rows),
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")
    os.chmod(path, 0o600)


def _stage_and_commit_outputs(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    manifest: dict[str, object],
) -> Path:
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"output directory already exists; choose a new path: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    os.chmod(staging, 0o700)
    try:
        for name, frame in tables.items():
            _write_frame(frame, staging / name)
        manifest["output_sha256"] = {
            name: sha256_file(staging / name) for name in sorted(tables)
        }
        manifest_path = staging / "analysis_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o600)
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir


def run_pipeline(config: AnalysisConfig) -> Path:
    _require(config.analysis in {"binary", "landmark", "both"}, "analysis must be binary, landmark, or both")
    lock = _read_json(config.feature_lock, "feature lock")
    features, settings = validate_feature_lock(lock)
    validate_provenance_chain(
        lock, config.structural_analysis_lock, config.phenotype_provenance
    )
    authorization = _read_json(config.authorization, "authorization record")
    validate_authorization(
        authorization,
        lock,
        config.feature_lock,
        config.outcomes,
        config.phenotypes,
        config.analysis,
        config.acknowledge_authorized_linkage,
    )

    outcomes = _read_csv_with_id(config.outcomes, "outcomes", ("recurrence_2y",))
    outcomes, has_dates = validate_outcomes(
        outcomes,
        config.analysis,
        int(lock["expected_patient_count"]),
        int(settings["binary_horizon_days"]),
        int(settings["blanking_period_days"]),
    )
    phenotypes = _read_csv_with_id(config.phenotypes, "phenotypes")
    merged, transformed = validate_and_merge_tables(
        outcomes, phenotypes, features, int(lock["expected_patient_count"])
    )
    development_mask, holdout_mask = partition_indices(merged, lock["partition"])
    if lock["source_doi"] == "10.5061/dryad.kkwh70sg0":
        development_recurrences = int(merged.loc[development_mask, "recurrence_2y"].sum())
        holdout_recurrences = int(merged.loc[holdout_mask, "recurrence_2y"].sum())
        _require(
            (development_recurrences, holdout_recurrences) == (40, 8),
            "linked recurrence counts do not reproduce the published 40/67 development and 8/15 holdout totals",
        )
    if config.analysis in {"binary", "both"}:
        development_outcome = merged.loc[development_mask, "recurrence_2y"].to_numpy(dtype=int)
        holdout_outcome = merged.loc[holdout_mask, "recurrence_2y"].to_numpy(dtype=int)
        minimum_binary_class = max(5, 3 * len(features))
        _require(
            min(int(development_outcome.sum()), len(development_outcome) - int(development_outcome.sum()))
            >= minimum_binary_class,
            f"the largest locked binary model requires at least {minimum_binary_class} development patients in each class",
        )
        _require(
            min(int(holdout_outcome.sum()), len(holdout_outcome) - int(holdout_outcome.sum())) >= 3,
            "the fixed holdout requires at least three patients in each binary outcome class",
        )
    if config.analysis in {"landmark", "both"}:
        _require(has_dates, "landmark analysis requested without exact dates and censoring information")

    tables: dict[str, pd.DataFrame] = {}
    if config.analysis in {"binary", "both"}:
        tables.update(fit_binary_models(
            merged,
            transformed,
            features,
            development_mask,
            holdout_mask,
            int(settings["bootstrap_replicates"]),
            int(settings["bootstrap_seed"]),
            int(settings["permutation_replicates"]),
            int(settings["permutation_seed"]),
        ))
        if lock["source_doi"] == "10.5061/dryad.kkwh70sg0":
            exact_assignments = int(
                tables["binary_holdout_primary_exact_randomization.csv"]
                .iloc[0]["number_fixed_count_assignments"]
            )
            _require(exact_assignments == 6435,
                     "Dryad holdout exact randomization must enumerate C(15,8)=6435 assignments")
    if config.analysis in {"landmark", "both"}:
        tables.update(fit_landmark_models(
            merged,
            transformed,
            features,
            development_mask,
            holdout_mask,
            float(settings["cox_ridge"]),
            int(settings["bootstrap_replicates"]),
            int(settings["bootstrap_seed"]),
        ))

    code_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": config.analysis,
        "cohort_name": lock["cohort_name"],
        "source_doi": lock["source_doi"],
        "n_linked_patients": len(merged),
        "n_recurrence_2y": int(merged["recurrence_2y"].sum()),
        "n_development_patients": int(np.sum(development_mask)),
        "n_holdout_patients": int(np.sum(holdout_mask)),
        "n_development_recurrence_2y": int(merged.loc[development_mask, "recurrence_2y"].sum()),
        "n_holdout_recurrence_2y": int(merged.loc[holdout_mask, "recurrence_2y"].sum()),
        "primary_evaluation": LOCKED_PRIMARY_EVALUATION,
        "secondary_evaluation": LOCKED_CROSS_FITTING,
        "complete_cohort_required": True,
        "structural_qc_policy": "P3 stops unless all 82 patients have a complete locked four-PV phenotype.",
        "holdout_fit_use": "none; all transformations and coefficients were estimated in development patients",
        "exact_dates_available": has_dates,
        "input_sha256": {
            "outcomes": sha256_file(config.outcomes),
            "phenotypes": sha256_file(config.phenotypes),
            "feature_lock": sha256_file(config.feature_lock),
            "structural_analysis_lock": sha256_file(config.structural_analysis_lock),
            "phenotype_provenance": sha256_file(config.phenotype_provenance),
            "authorization_record": sha256_file(config.authorization),
            "analysis_code": sha256_file(code_path),
        },
        "locked_features": [feature.__dict__ for feature in features],
        "locked_analysis": dict(settings),
        "binary_interval_scope": "patient bootstrap in the fixed holdout; development-only LOO is secondary",
        "bootstrap_refits_training_models": False,
        "bootstrap_scope_caveat": "Intervals resample fixed predictions within the named partition and exclude training-fit uncertainty.",
        "holdout_randomization_scope": "Exact fixed-event-count label enumeration with development-fitted M0/M1 predictions held fixed.",
        "landmark_time_origin": "post_ablation_mri_date",
        "landmark_administrative_end": "730 days after ablation or earlier verified loss to follow-up",
        "landmark_entry_rule": "qualifying-recurrence-free at post_ablation_mri_date",
        "interpretation_boundary": "Fixed-cohort association only; no causal claim and no validation beyond the published holdout.",
    }
    return _stage_and_commit_outputs(config.output_dir, tables, manifest)


def _expect_failure(function: Callable[[], object], exception: type[BaseException], label: str) -> None:
    try:
        function()
    except exception:
        return
    raise AssertionError(f"self-test expected {exception.__name__}: {label}")


def run_self_test() -> None:
    """Exercise positive and negative paths entirely inside a temporary directory."""
    intercept_outcome = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    intercept_beta, _ = fit_firth_logistic(np.empty((len(intercept_outcome), 0)), intercept_outcome)
    expected_probability = (float(intercept_outcome.sum()) + 0.5) / (len(intercept_outcome) + 1.0)
    if not np.isclose(expit(intercept_beta[0]), expected_probability, rtol=1.0e-8, atol=1.0e-10):
        raise AssertionError("Firth intercept does not match its closed-form bias-reduced estimate")
    separated_x = np.r_[np.linspace(-2.0, -0.2, 12), np.linspace(0.2, 2.0, 12)][:, None]
    separated_y = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
    separated_beta, _ = fit_firth_logistic(separated_x, separated_y)
    separated_design = np.column_stack([np.ones(len(separated_y)), separated_x])
    _, separated_score, _ = _firth_components(separated_design, separated_y, separated_beta)
    _require_stationary(separated_score, "self-test separated Firth fit")
    if not np.isfinite(separated_beta).all():
        raise AssertionError("Firth separated-data coefficients are not finite")
    _, false_bound_score, _ = _firth_components(
        separated_design, separated_y, np.array([0.0, 25.0])
    )
    _expect_failure(
        lambda: _require_stationary(false_bound_score, "fabricated bound-reported optimizer success"),
        ModelFitError,
        "nonstationary success at a coefficient bound",
    )
    tied_outcome = np.array([1, 0, 0, 1, 0], dtype=int)
    if not np.isclose(_average_precision(tied_outcome, np.full(5, 0.4)), 0.4):
        raise AssertionError("tie-aware average precision does not reduce to prevalence for a constant score")

    project_root = Path(__file__).resolve().parents[1]
    dryad_lock = json.loads(
        (project_root / "templates" / "recurrence_feature_lock.template.json").read_text(encoding="utf-8")
    )
    dryad_lock["locked"] = True
    dryad_lock["locked_at_utc"] = "2026-01-01T08:00:00Z"
    dryad_lock["outcome_information_used"] = False
    dryad_lock["structural_analysis_lock_sha256"] = "a" * 64
    dryad_lock["phenotype_provenance_sha256"] = "b" * 64
    dryad_lock["phenotype_table_sha256"] = "e" * 64
    validate_feature_lock(dryad_lock)
    altered_dryad_lock = json.loads(json.dumps(dryad_lock))
    altered_dryad_lock["partition"]["holdout_ids"][0] = "ID016"
    _expect_failure(
        lambda: validate_feature_lock(altered_dryad_lock),
        DataValidationError,
        "altered Dryad holdout split",
    )
    boolean_numeric_lock = json.loads(json.dumps(dryad_lock))
    boolean_numeric_lock["analysis"]["cox_ridge"] = True
    _expect_failure(
        lambda: validate_feature_lock(boolean_numeric_lock),
        DataValidationError,
        "JSON boolean used as numeric Cox penalty",
    )

    with tempfile.TemporaryDirectory(prefix="recurrence-pipeline-self-test-") as directory:
        root = Path(directory)
        rng = np.random.default_rng(20260903)
        n = 48
        patient_ids = [f"SYN{i:03d}" for i in range(n)]
        capacity = np.clip(rng.normal(0.12, 0.035, n), 0.02, 0.25)
        gap_fraction = np.clip(rng.beta(2.0, 8.0, n), 0.01, 0.75)
        age = rng.normal(64.0, 8.0, n)
        risk = 7.0 * capacity + 2.5 * gap_fraction + 0.01 * (age - 64.0) + rng.normal(0.0, 0.3, n)
        development_ids = patient_ids[:36]
        holdout_ids = patient_ids[36:]
        event_indices = set(np.argsort(risk[:36])[-12:])
        event_indices.update(36 + index for index in np.argsort(risk[36:])[-4:])
        ablation_dates: list[str] = []
        mri_dates: list[str] = []
        followup_dates: list[str] = []
        recurrence_dates: list[str] = []
        labels: list[int] = []
        early_count = 0
        for index in range(n):
            ablation = pd.Timestamp("2020-01-01") + pd.Timedelta(days=index)
            mri = ablation + pd.Timedelta(days=120)
            followup = ablation + pd.Timedelta(days=820)
            if index in event_indices:
                event_day = 100 if early_count < 2 else 190 + (index % 20) * 7
                early_count += 1
                recurrence = ablation + pd.Timedelta(days=event_day)
                recurrence_dates.append(recurrence.strftime("%Y-%m-%d"))
                labels.append(1)
            else:
                recurrence_dates.append("")
                labels.append(0)
            ablation_dates.append(ablation.strftime("%Y-%m-%d"))
            mri_dates.append(mri.strftime("%Y-%m-%d"))
            followup_dates.append(followup.strftime("%Y-%m-%d"))

        outcomes = pd.DataFrame({
            "patient_id": patient_ids,
            "recurrence_2y": labels,
            "ablation_date": ablation_dates,
            "post_ablation_mri_date": mri_dates,
            "followup_end_date": followup_dates,
            "recurrence_date": recurrence_dates,
        })
        phenotypes = pd.DataFrame({
            "patient_id": patient_ids,
            "max_pv_normalized_capacity": capacity,
            "total_gap_arc_fraction": gap_fraction,
            "age_years": age,
        })
        blank_date_binary = outcomes[["patient_id", "recurrence_2y"]].copy()
        for column in DATE_COLUMNS:
            blank_date_binary[column] = ""
        _, blank_dates_available = validate_outcomes(
            blank_date_binary, "binary", n, 730, 90
        )
        if blank_dates_available:
            raise AssertionError("uniformly blank date columns were treated as observed dates")
        _expect_failure(
            lambda: validate_outcomes(blank_date_binary, "landmark", n, 730, 90),
            DataValidationError,
            "landmark request with uniformly blank date columns",
        )
        invalid_label_lexeme = outcomes[["patient_id", "recurrence_2y"]].copy()
        invalid_label_lexeme["recurrence_2y"] = invalid_label_lexeme["recurrence_2y"].astype(str)
        invalid_label_lexeme.loc[0, "recurrence_2y"] = "1.0"
        _expect_failure(
            lambda: validate_outcomes(invalid_label_lexeme, "binary", n, 730, 90),
            DataValidationError,
            "non-integer recurrence CSV lexeme",
        )
        lock = {
            "lock_version": "1.0",
            "locked": True,
            "locked_at_utc": "2026-01-01T09:00:00Z",
            "outcome_information_used": False,
            "cohort_name": "synthetic self-test cohort",
            "source_doi": "self-test-doi",
            "structural_analysis_lock_sha256": "c" * 64,
            "phenotype_provenance_sha256": "d" * 64,
            "phenotype_table_sha256": "e" * 64,
            "expected_patient_count": n,
            "id_column": "patient_id",
            "partition": {
                "source": "synthetic fixed split",
                "development_ids": development_ids,
                "holdout_ids": holdout_ids,
            },
            "phenotype_definition": {
                "primary_feature": "max_pv_normalized_capacity",
                "symbol": "synthetic_feature",
                "estimator": "synthetic generator",
                "capacity_per_rotation": "synthetic_capacity",
                "conductivity_map": "synthetic_positive_map",
                "full_reference_score": "synthetic_signed_reference",
                "reconstruction_score": "synthetic_unclipped_state",
                "probability_clipping_only": True,
                "within_pv_rotation_aggregation": "median",
                "across_pv_aggregation": "maximum",
                "operation_order": ["generate", "aggregate", "transform"],
                "definition": "synthetic identity check",
                "log_offset": 0.0,
                "strict_positive_capacity_required": True,
                "higher_means": "higher synthetic value",
                "outcome_blind": True,
            },
            "features": [
                {
                    "name": "max_pv_normalized_capacity",
                    "role": "phenotype",
                    "transform": "identity",
                    "units": "dimensionless",
                    "direction": "higher indicates greater passive leakage",
                    "description": "Synthetic maximum per-vein capacity.",
                },
                {
                    "name": "total_gap_arc_fraction",
                    "role": "phenotype",
                    "transform": "identity",
                    "units": "fraction",
                    "direction": "higher indicates more reconstructed gap arc",
                    "description": "Synthetic total gap fraction.",
                },
                {
                    "name": "age_years",
                    "role": "clinical",
                    "transform": "identity",
                    "units": "years",
                    "direction": "higher indicates older age",
                    "description": "Synthetic adjustment variable.",
                },
            ],
            "analysis": {
                "binary_horizon_days": 730,
                "blanking_period_days": 90,
                "primary_evaluation": LOCKED_PRIMARY_EVALUATION,
                "cross_fitting": LOCKED_CROSS_FITTING,
                "standardization": LOCKED_STANDARDIZATION,
                "binary_estimator": LOCKED_BINARY_ESTIMATOR,
                "survival_estimator": LOCKED_SURVIVAL_ESTIMATOR,
                "cox_ridge": 0.1,
                "bootstrap_replicates": 200,
                "bootstrap_seed": 20260903,
                "permutation_replicates": 20,
                "permutation_seed": 20260904,
                "complete_cohort_required": True,
                "missing_predictors": "fail",
                "max_total_predictors": 4,
            },
        }
        outcomes_path = root / "outcomes.csv"
        phenotypes_path = root / "phenotypes.csv"
        lock_path = root / "feature_lock.json"
        structural_lock_path = root / "structural_analysis_lock.json"
        phenotype_provenance_path = root / "phenotype_provenance.json"
        authorization_path = root / "authorization.json"
        outcomes.to_csv(outcomes_path, index=False)
        phenotypes.to_csv(phenotypes_path, index=False)
        structural_lock = {
            "schema_version": "1.0",
            "status": "LOCKED",
            "frozen_phenotype": _expected_structural_phenotype(lock["phenotype_definition"]),
        }
        structural_lock_path.write_text(json.dumps(structural_lock, indent=2) + "\n", encoding="utf-8")
        lock["structural_analysis_lock_sha256"] = sha256_file(structural_lock_path)
        phenotype_provenance = {
            "schema_version": "1.0",
            "scope": "complete_locked_cohort",
            "n_patients": n,
            "feature": lock["phenotype_definition"]["primary_feature"],
            "definition": lock["phenotype_definition"]["definition"],
            "analysis_lock_sha256": lock["structural_analysis_lock_sha256"],
            "phenotype_table_sha256": sha256_file(phenotypes_path),
            "long_form_phenotype_sha256": "a" * 64,
            "run_provenance_sha256": "c" * 64,
            "outcomes_accessed": False,
        }
        phenotype_provenance_path.write_text(
            json.dumps(phenotype_provenance, indent=2) + "\n", encoding="utf-8"
        )
        lock["phenotype_provenance_sha256"] = sha256_file(phenotype_provenance_path)
        lock["phenotype_table_sha256"] = sha256_file(phenotypes_path)
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        authorization = {
            "authorization_version": "1.0",
            "data_use_authorized": True,
            "id_linkage_authorized": True,
            "deidentified_or_pseudonymized": True,
            "outcomes_publicly_available": False,
            "recurrence_definition_verified": True,
            "feature_definitions_locked_before_linkage": True,
            "recurrence_date_blank_means_censored": True,
            "authorized_by": "synthetic self-test",
            "authorized_at_utc": "2026-01-01T09:30:00Z",
            "outcome_linkage_at_utc": "2026-01-01T10:00:00Z",
            "data_use_basis": "synthetic self-test data only",
            "outcome_definition": "first qualifying recurrence after a 90-day blanking period",
            "minimum_duration_seconds": 30,
            "blanking_period_days": 90,
            "binary_horizon_days": 730,
            "eligible_rhythms": ["atrial_fibrillation", "atrial_flutter"],
            "cohort_name": lock["cohort_name"],
            "source_doi": lock["source_doi"],
            "id_linkage_key": "patient_id",
            "authorized_analyses": ["binary_2y", "post_mri_landmark"],
            "feature_lock_sha256": sha256_file(lock_path),
            "outcomes_sha256": sha256_file(outcomes_path),
            "phenotypes_sha256": sha256_file(phenotypes_path),
        }
        authorization_path.write_text(json.dumps(authorization, indent=2) + "\n", encoding="utf-8")
        config = AnalysisConfig(
            outcomes_path,
            phenotypes_path,
            authorization_path,
            lock_path,
            structural_lock_path,
            phenotype_provenance_path,
            "both",
            root / "results",
            True,
        )
        result = run_pipeline(config)
        expected_outputs = {
            "binary_holdout_patient_predictions.csv",
            "binary_holdout_metrics.csv",
            "binary_holdout_model_comparisons.csv",
            "binary_holdout_primary_exact_randomization.csv",
            "binary_development_loo_predictions.csv",
            "binary_development_loo_metrics.csv",
            "binary_development_loo_model_comparisons.csv",
            "binary_development_coefficients.csv",
            "binary_development_primary_permutation_draws.csv",
            "binary_development_primary_permutation_summary.csv",
            "landmark_eligibility.csv",
            "landmark_holdout_patient_scores.csv",
            "landmark_holdout_metrics.csv",
            "landmark_holdout_model_comparisons.csv",
            "landmark_development_loo_scores.csv",
            "landmark_development_loo_metrics.csv",
            "landmark_development_loo_model_comparisons.csv",
            "landmark_development_coefficients.csv",
            "analysis_manifest.json",
        }
        if {path.name for path in result.iterdir()} != expected_outputs:
            raise AssertionError("self-test result inventory is incomplete")
        predictions = pd.read_csv(result / "binary_holdout_patient_predictions.csv")
        if len(predictions) != len(holdout_ids) or not bool(np.isfinite(predictions.filter(like="probability_").to_numpy(dtype=float)).all()):
            raise AssertionError("self-test binary predictions are invalid")
        exact_randomization = pd.read_csv(
            result / "binary_holdout_primary_exact_randomization.csv"
        ).iloc[0]
        if int(exact_randomization["number_fixed_count_assignments"]) != math.comb(12, 4):
            raise AssertionError("self-test exact holdout enumeration count is incorrect")
        if not 0.0 < float(exact_randomization["exact_one_sided_p_negative_delta"]) <= 1.0:
            raise AssertionError("self-test exact holdout p-value is outside (0,1]")
        eligibility = pd.read_csv(result / "landmark_eligibility.csv")
        if int((eligibility["landmark_eligible"] == 0).sum()) != 2:
            raise AssertionError("self-test landmark exclusions are incorrect")

        binary_only_outcomes_path = root / "binary_only_outcomes.csv"
        outcomes[["patient_id", "recurrence_2y"]].to_csv(binary_only_outcomes_path, index=False)
        binary_only_authorization = dict(authorization)
        binary_only_authorization["outcomes_sha256"] = sha256_file(binary_only_outcomes_path)
        binary_only_authorization_path = root / "binary_only_authorization.json"
        binary_only_authorization_path.write_text(
            json.dumps(binary_only_authorization, indent=2) + "\n", encoding="utf-8"
        )
        binary_only_config = AnalysisConfig(
            binary_only_outcomes_path,
            phenotypes_path,
            binary_only_authorization_path,
            lock_path,
            structural_lock_path,
            phenotype_provenance_path,
            "binary",
            root / "binary-only-results",
            True,
        )
        binary_only_result = run_pipeline(binary_only_config)
        if (binary_only_result / "landmark_holdout_metrics.csv").exists():
            raise AssertionError("binary-only self-test unexpectedly wrote a landmark output")

        permuted_binary_outcomes = outcomes[["patient_id", "recurrence_2y"]].copy()
        holdout_rows = permuted_binary_outcomes["patient_id"].isin(holdout_ids)
        permuted_binary_outcomes.loc[holdout_rows, "recurrence_2y"] = np.roll(
            permuted_binary_outcomes.loc[holdout_rows, "recurrence_2y"].to_numpy(), 3
        )
        permuted_path = root / "permuted_holdout_outcomes.csv"
        permuted_binary_outcomes.to_csv(permuted_path, index=False)
        permuted_authorization = dict(authorization)
        permuted_authorization["outcomes_sha256"] = sha256_file(permuted_path)
        permuted_authorization_path = root / "permuted_holdout_authorization.json"
        permuted_authorization_path.write_text(
            json.dumps(permuted_authorization, indent=2) + "\n", encoding="utf-8"
        )
        permuted_config = AnalysisConfig(
            permuted_path,
            phenotypes_path,
            permuted_authorization_path,
            lock_path,
            structural_lock_path,
            phenotype_provenance_path,
            "binary",
            root / "permuted-holdout-results",
            True,
        )
        permuted_result = run_pipeline(permuted_config)
        original_probabilities = pd.read_csv(
            binary_only_result / "binary_holdout_patient_predictions.csv"
        ).filter(like="probability_").to_numpy(dtype=float)
        permuted_probabilities = pd.read_csv(
            permuted_result / "binary_holdout_patient_predictions.csv"
        ).filter(like="probability_").to_numpy(dtype=float)
        if not np.array_equal(original_probabilities, permuted_probabilities):
            raise AssertionError("holdout outcomes changed fixed model probabilities")

        no_date_landmark_config = AnalysisConfig(
            binary_only_outcomes_path,
            phenotypes_path,
            binary_only_authorization_path,
            lock_path,
            structural_lock_path,
            phenotype_provenance_path,
            "landmark",
            root / "no-date-landmark-results",
            True,
        )
        _expect_failure(
            lambda: run_pipeline(no_date_landmark_config),
            DataValidationError,
            "landmark analysis without dates",
        )

        unauthorised = AnalysisConfig(
            outcomes_path,
            phenotypes_path,
            authorization_path,
            lock_path,
            structural_lock_path,
            phenotype_provenance_path,
            "binary",
            root / "unauthorised-results",
            False,
        )
        _expect_failure(lambda: run_pipeline(unauthorised), AuthorizationError, "missing explicit acknowledgement")

        bad_outcomes = outcomes.copy()
        bad_outcomes.loc[0, "post_ablation_mri_date"] = "2019-12-31"
        bad_outcomes_path = root / "bad_outcomes.csv"
        bad_outcomes.to_csv(bad_outcomes_path, index=False)
        bad_authorization = dict(authorization)
        bad_authorization["outcomes_sha256"] = sha256_file(bad_outcomes_path)
        bad_authorization_path = root / "bad_authorization.json"
        bad_authorization_path.write_text(json.dumps(bad_authorization, indent=2) + "\n", encoding="utf-8")
        bad_config = AnalysisConfig(
            bad_outcomes_path,
            phenotypes_path,
            bad_authorization_path,
            lock_path,
            structural_lock_path,
            phenotype_provenance_path,
            "both",
            root / "bad-results",
            True,
        )
        _expect_failure(lambda: run_pipeline(bad_config), DataValidationError, "impossible MRI chronology")

        wrong_hash = dict(authorization)
        wrong_hash["phenotypes_sha256"] = "0" * 64
        wrong_hash_path = root / "wrong_hash_authorization.json"
        wrong_hash_path.write_text(json.dumps(wrong_hash, indent=2) + "\n", encoding="utf-8")
        wrong_hash_config = AnalysisConfig(
            outcomes_path,
            phenotypes_path,
            wrong_hash_path,
            lock_path,
            structural_lock_path,
            phenotype_provenance_path,
            "binary",
            root / "wrong-hash-results",
            True,
        )
        _expect_failure(lambda: run_pipeline(wrong_hash_config), AuthorizationError, "mismatched phenotype hash")

        wrong_provenance_path = root / "wrong_phenotype_provenance.json"
        wrong_provenance_path.write_text('{"synthetic_phenotype_provenance":false}\n', encoding="utf-8")
        wrong_provenance_config = AnalysisConfig(
            outcomes_path,
            phenotypes_path,
            authorization_path,
            lock_path,
            structural_lock_path,
            wrong_provenance_path,
            "binary",
            root / "wrong-provenance-results",
            True,
        )
        _expect_failure(
            lambda: run_pipeline(wrong_provenance_config),
            DataValidationError,
            "mismatched upstream phenotype provenance hash",
        )
    print("recurrence pipeline self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run synthetic positive and negative tests in a temporary directory")
    parser.add_argument("--hash-file", type=Path, help="print the SHA-256 digest of one file and exit")
    parser.add_argument("--outcomes", type=Path, help="authorised ID-linked recurrence CSV")
    parser.add_argument("--phenotypes", type=Path, help="outcome-free ID-linked phenotype CSV")
    parser.add_argument("--authorization", type=Path, help="completed linkage authorisation JSON")
    parser.add_argument("--feature-lock", type=Path, help="completed outcome-blind feature-lock JSON")
    parser.add_argument("--structural-analysis-lock", type=Path, help="frozen upstream structural-analysis lock")
    parser.add_argument("--phenotype-provenance", type=Path, help="outcome-free combined phenotype provenance manifest")
    parser.add_argument("--analysis", choices=["binary", "landmark", "both"], help="prespecified analysis path")
    parser.add_argument("--output-dir", type=Path, help="new, non-existing directory for linked results")
    parser.add_argument(
        "--acknowledge-authorized-linkage",
        action="store_true",
        help="confirm that the caller is authorised to link these exact pseudonymous ID tables",
    )
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.self_test:
        if any(value is not None for value in (
            arguments.hash_file,
            arguments.outcomes,
            arguments.phenotypes,
            arguments.authorization,
            arguments.feature_lock,
            arguments.structural_analysis_lock,
            arguments.phenotype_provenance,
            arguments.analysis,
            arguments.output_dir,
        )) or arguments.acknowledge_authorized_linkage:
            parser.error("--self-test cannot be combined with analysis inputs")
        run_self_test()
        return
    if arguments.hash_file is not None:
        if any(value is not None for value in (
            arguments.outcomes,
            arguments.phenotypes,
            arguments.authorization,
            arguments.feature_lock,
            arguments.structural_analysis_lock,
            arguments.phenotype_provenance,
            arguments.analysis,
            arguments.output_dir,
        )) or arguments.acknowledge_authorized_linkage:
            parser.error("--hash-file cannot be combined with analysis inputs")
        if not arguments.hash_file.is_file():
            parser.error(f"file does not exist: {arguments.hash_file}")
        print(sha256_file(arguments.hash_file))
        return

    required = {
        "--outcomes": arguments.outcomes,
        "--phenotypes": arguments.phenotypes,
        "--authorization": arguments.authorization,
        "--feature-lock": arguments.feature_lock,
        "--structural-analysis-lock": arguments.structural_analysis_lock,
        "--phenotype-provenance": arguments.phenotype_provenance,
        "--analysis": arguments.analysis,
        "--output-dir": arguments.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    config = AnalysisConfig(
        arguments.outcomes,
        arguments.phenotypes,
        arguments.authorization,
        arguments.feature_lock,
        arguments.structural_analysis_lock,
        arguments.phenotype_provenance,
        arguments.analysis,
        arguments.output_dir,
        arguments.acknowledge_authorized_linkage,
    )
    try:
        result = run_pipeline(config)
    except RecurrencePipelineError as error:
        parser.exit(2, f"recurrence analysis stopped: {error}\n")
    print(f"recurrence analysis completed: {result}")


if __name__ == "__main__":
    main()
