"""Current synthetic applied experiments for the sign-graph manuscript.

The publication driver has two parts.

1. A geometry-replicated stress test crosses a complete ring, narrow, wide,
   double and oblique gaps with four fixed spatial blackouts.  Passive and graph
   flows receive the same 120-step pseudo-time horizon; block repeats are
   averaged before geometry-level contrasts and continuous-score calibration.
2. A width--residual-diffusivity study uses subcell-averaged lesion profiles,
   harmonic finite-volume transmissibilities, several gap angles relative to a
   fixed fibre direction, an annular effective-conductance functional, and
   bidirectional pacing on selected cases.

All synthetic truth fields are used only to generate observations and evaluate
completed reconstructions.  They are never supplied to the reconstruction solve.
Superseded single-gap horizon-selection and synthetic end-to-end functions are
retained below solely for provenance and are unreachable from the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator, cg

from model import (
    EPParameters,
    PhaseParameters,
    SpectralGrid,
    annular_capacity,
    diffusivity_factor,
    ring_gap_width,
    segmentation_metrics,
    solve_monodomain,
    solve_phase,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.size": 8.3,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.3,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "mathtext.fontset": "stix",
    }
)


# Geometry and reconstruction parameters.
LENGTH = 60.0  # mm
CENTER = (30.0, 30.0)
RADIUS = 15.0
LESION_WIDTH = 4.0
TRANSITION = 1.50
TRUTH_GAP_WIDTH = 4.50
TRUTH_GAP_ANGLE = np.deg2rad(45.0)
TRUTH_GAP_SCORE = -1.0

RECON_N = 81  # odd: no singled-out Fourier Nyquist mode
RECON_KX = 1.0
RECON_KY = 0.60
PHASE_DT = 0.01
TERMINAL_WINDOW = 12
CANDIDATE_HORIZONS = (0.15, 0.30, 0.60, 1.20, 2.40, 4.80)
COMMON_HORIZONS = (0.60, 1.20)
COMMON_HORIZON_SUMMARY_METRICS = (
    "rmse",
    "capacity_abs_error",
    "gap_width_abs_error_mm",
    "gap_region_rmse",
)
COMMON_HORIZON_BOOTSTRAP_SEED = 20260903
COMMON_HORIZON_BOOTSTRAP_RESAMPLES = 5000

# Geometry-replicated missing-sector stress test.  The four acquisition blocks
# are fixed in anatomical coordinates and are used for every truth geometry;
# they are neither inferred from nor centred adaptively on a gap.  A common
# 120-step horizon is imposed on passive and graph flows, so this experiment
# makes no data-dependent pseudo-time choice.
GEOMETRY_RECONSTRUCTION_HORIZON = 1.20
GEOMETRY_BLACKOUT_ANGLES_DEG = (0.0, 90.0, 180.0, 270.0)
GEOMETRY_ACQUISITION_SEEDS = (3101, 3103, 3107, 3109)
GEOMETRY_SPECS = (
    {
        "geometry": "complete_ring",
        "gaps": (),
        "description": "complete ring",
    },
    {
        "geometry": "narrow_gap",
        "gaps": ((3.0, 0.0, 0.0),),
        "description": "one 3-mm gap",
    },
    {
        "geometry": "wide_gap",
        "gaps": ((9.0, 90.0, 0.0),),
        "description": "one 9-mm gap",
    },
    {
        "geometry": "two_gaps",
        "gaps": ((4.5, 0.0, 0.0), (7.0, 180.0, 0.0)),
        "description": "4.5- and 7-mm gaps",
    },
    {
        "geometry": "oblique_gap",
        "gaps": ((6.0, 270.0, 1.4),),
        "description": "one 6-mm oblique gap",
    },
)
GEOMETRY_SUMMARY_METRICS = (
    "masked_score_rmse",
    "masked_scaled_score_mse",
    "masked_calibration_ece",
    "masked_calibration_slope_abs_error",
    "capacity_abs_error",
    "gap_count_abs_error",
    "largest_gap_width_abs_error_mm",
)

# Compact observations and held-out region.
KERNEL_SUPPORT = 2.25  # mm
KERNEL_WEIGHT = 15.0
BLACKOUT_ANGLE = np.deg2rad(45.0)  # declared acquisition dropout, not inferred truth
BLACKOUT_ARC_WIDTH = 12.0  # mm, independently predeclared missing-contact sector
HOLDOUT_BUFFER = 1.00  # mm around the declared blackout sector
VALIDATION_FRACTION = 0.20
CONTACT_NOISE_SD = 0.12

# Score-to-diffusivity law used only for reconstructed score fields.
ETA_MIN = 1.0e-3
ETA_SLOPE = 8.0

CALIBRATION_SEEDS = (101, 103, 107, 109)
TEST_SEEDS = (3, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)
REPRESENTATIVE_SEED = 11
EP_ENSEMBLE_SEEDS = (3, 11, 23, 31, 41, 47, 59, 67)
PRODUCTION_EP_N = 121
PRODUCTION_EP_DT = 0.02
TARGET_SECTOR_HALF_ANGLE = np.deg2rad(15.0)
TARGET_CAPTURE_FRACTION = 0.80
STIMULUS_RADIUS = 3.5
STIMULUS_AMPLITUDE = 1.2
STIMULUS_DURATION = 2.0
SUBCELLS_PER_AXIS = 3

# Files that constitute the current numerical publication set.  This list is
# deliberately explicit: retired single-gap ensembles and the superseded
# synthetic end-to-end experiment must not become current merely because an
# old file is still present in ``data`` or ``figures``.
PUBLICATION_ARTIFACT_PATHS = (
    "requirements.txt",
    "code/model.py",
    "code/fetch_zenodo_erp_subset.py",
    "code/check_patient_contact_geometry.py",
    "code/legacy_vtk_polydata.py",
    "code/patient_method_checkpoint.py",
    "code/patient_surface_plot.py",
    "code/run_applied.py",
    "code/run_core.py",
    "code/run_patient_reconstruction.py",
    "code/run_surface_verification.py",
    "code/run_zenodo_pvi_reconstruction.py",
    "code/surface_fem.py",
    "code/surface_mesh.py",
    "code/surface_phase.py",
    "code/test_surface_fem.py",
    "code/test_surface_phase.py",
    "code/test_fetch_zenodo_erp_subset.py",
    "code/test_legacy_vtk_polydata.py",
    "code/test_patient_method_checkpoint.py",
    "code/validate_outputs.py",
    "code/validate_zenodo_pvi_outputs.py",
    "data/conduction_calibration.csv",
    "data/ep_exact_spatial.csv",
    "data/ep_exact_temporal.csv",
    "data/graph_limit.csv",
    "data/graph_representative.npz",
    "data/phase_exact_solution_spatial.csv",
    "data/phase_exact_solution_spatial_time_control.json",
    "data/phase_exact_spatial.csv",
    "data/phase_exact_temporal.csv",
    "data/phase_exact_term_balance.json",
    "data/sparse_reconstruction_geometry_blocks.csv",
    "data/sparse_reconstruction_geometry_units.csv",
    "data/sparse_reconstruction_geometry_summary.csv",
    "data/sparse_reconstruction_geometry_calibration.csv",
    "data/sparse_reconstruction_geometry_maps.npz",
    "data/pvi_capacity_phase_diagram.csv",
    "data/pvi_bidirectional_ep.csv",
    "data/pvi_capacity_refinement.csv",
    "data/pvi_capacity_representatives.npz",
    "data/zenodo_pvi_mesh_inventory.csv",
    "data/zenodo_pvi_boundary_transfer.csv",
    "data/zenodo_pvi_reconstruction_metrics.csv",
    "data/zenodo_pvi_capacity.csv",
    "data/zenodo_pvi_patient_summary.csv",
    "data/zenodo_pvi_patient_contrasts.csv",
    "data/zenodo_pvi_representative.npz",
    "data/zenodo_pvi_provenance.json",
    "data/zenodo_pvi_output_schema.json",
    "data/zenodo_pvi_contact_geometry.csv",
    "figures/fig1_exact_verification.pdf",
    "figures/fig1_exact_verification.png",
    "figures/fig2_graph_limit.pdf",
    "figures/fig2_graph_limit.png",
    "figures/fig3_conduction_calibration.pdf",
    "figures/fig3_conduction_calibration.png",
    "figures/fig4_sparse_reconstruction.pdf",
    "figures/fig4_sparse_reconstruction.png",
    "figures/fig5_pvi_capacity_phase_diagram.pdf",
    "figures/fig5_pvi_capacity_phase_diagram.png",
    "figures/fig6_patient_surface.pdf",
    "figures/fig6_patient_surface.png",
)


def reconstruction_grid(n: int = RECON_N) -> SpectralGrid:
    return SpectralGrid(
        n,
        n,
        LENGTH,
        LENGTH,
        kx_weight=RECON_KX,
        ky_weight=RECON_KY,
    )


def _wrapped_angle(theta: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * theta))


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lesion_components(
    X: np.ndarray,
    Y: np.ndarray,
    gap_width: float,
    gap_angle: float,
    transition: float = TRANSITION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return radial band, arc-gap profile, radius and arc distance.

    The arc profile is one half at ``|q| = gap_width / 2``.  Consequently the
    prescribed gap width is the zero-level centreline width when the gap score
    is -1; it is not a narrower constant core with extra hidden shoulders.
    """
    rx = X - CENTER[0]
    ry = Y - CENTER[1]
    radius = np.sqrt(rx * rx + ry * ry)
    theta = np.arctan2(ry, rx)

    radial_distance = np.abs(radius - RADIUS)
    radial_coordinate = (radial_distance - 0.5 * LESION_WIDTH) / transition
    band = 1.0 - _smoothstep01(radial_coordinate)

    angular_distance = np.abs(_wrapped_angle(theta - gap_angle))
    arc_distance = RADIUS * angular_distance
    if gap_width <= 0.0:
        channel = np.zeros_like(radius)
    else:
        channel = 0.5 * (
            1.0 - np.tanh((arc_distance - 0.5 * gap_width) / transition)
        )
    return band, channel, radius, arc_distance


def lesion_score_field(
    X: np.ndarray,
    Y: np.ndarray,
    gap_width: float = TRUTH_GAP_WIDTH,
    gap_score: float = TRUTH_GAP_SCORE,
    gap_angle: float = TRUTH_GAP_ANGLE,
    transition: float = TRANSITION,
) -> np.ndarray:
    band, channel, _, _ = lesion_components(X, Y, gap_width, gap_angle, transition)
    return -1.0 + band * (2.0 - (1.0 - gap_score) * channel)


def geometry_score_field(
    X: np.ndarray,
    Y: np.ndarray,
    gaps: tuple[tuple[float, float, float], ...],
    transition: float = TRANSITION,
) -> np.ndarray:
    """Return a ring with zero, one, or several possibly oblique gaps.

    A gap tuple contains ``(zero_level_width_mm, centre_angle_deg,
    tangent_shift_per_radial_mm)``.  The last parameter shifts the gap centre
    tangentially across the lesion thickness.  Thus zero gives a radial gap,
    whereas 1.4 gives a visibly oblique crossing without changing the smooth
    transition law used elsewhere in the paper.
    """

    rx = X - CENTER[0]
    ry = Y - CENTER[1]
    radius = np.sqrt(rx * rx + ry * ry)
    theta = np.arctan2(ry, rx)
    radial_distance = np.abs(radius - RADIUS)
    radial_coordinate = (radial_distance - 0.5 * LESION_WIDTH) / transition
    band = 1.0 - _smoothstep01(radial_coordinate)
    if not gaps:
        channel = np.zeros_like(radius)
    else:
        channels = []
        for width_mm, angle_deg, tangent_shift in gaps:
            signed_arc = (
                RADIUS * _wrapped_angle(theta - np.deg2rad(angle_deg))
                - tangent_shift * (radius - RADIUS)
            )
            channels.append(
                0.5
                * (1.0 - np.tanh((np.abs(signed_arc) - 0.5 * width_mm) / transition))
            )
        channel = np.maximum.reduce(channels)
    return -1.0 + band * (2.0 - 2.0 * channel)


def lesion_diffusivity_field(
    X: np.ndarray,
    Y: np.ndarray,
    gap_width: float,
    gap_diffusivity: float,
    gap_angle: float,
    lesion_diffusivity: float = ETA_MIN,
    transition: float = TRANSITION,
) -> np.ndarray:
    """Continuous lesion profile parameterised directly by diffusivity fraction."""
    if not (0.0 < lesion_diffusivity <= gap_diffusivity <= 1.0):
        raise ValueError("require 0 < lesion diffusivity <= gap diffusivity <= 1")
    band, channel, _, _ = lesion_components(X, Y, gap_width, gap_angle, transition)
    return 1.0 - band * (
        (1.0 - lesion_diffusivity)
        - (gap_diffusivity - lesion_diffusivity) * channel
    )


def holdout_mask(
    grid: SpectralGrid,
    blackout_width: float = BLACKOUT_ARC_WIDTH,
    blackout_angle: float = BLACKOUT_ANGLE,
) -> np.ndarray:
    """Coverage mask imposed after compact-kernel accumulation.

    Compact Wendland support already localises each observation.  This separate
    mask guarantees an exactly zero-confidence gap and buffer, including at its
    boundary where a neighbouring compact kernel could otherwise overlap.
    """
    _, _, radius, arc_distance = lesion_components(
        grid.X, grid.Y, blackout_width, blackout_angle, TRANSITION
    )
    radial_limit = 0.5 * LESION_WIDTH + HOLDOUT_BUFFER
    arc_limit = 0.5 * blackout_width + HOLDOUT_BUFFER
    return (np.abs(radius - RADIUS) <= radial_limit) & (arc_distance <= arc_limit)


def _points_in_holdout(
    x: np.ndarray,
    y: np.ndarray,
    blackout_width: float = BLACKOUT_ARC_WIDTH,
    blackout_angle: float = BLACKOUT_ANGLE,
) -> np.ndarray:
    radius = np.sqrt((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2)
    theta = np.arctan2(y - CENTER[1], x - CENTER[0])
    arc_distance = RADIUS * np.abs(_wrapped_angle(theta - blackout_angle))
    radial_limit = 0.5 * LESION_WIDTH + HOLDOUT_BUFFER + KERNEL_SUPPORT
    arc_limit = 0.5 * blackout_width + HOLDOUT_BUFFER + KERNEL_SUPPORT
    return (np.abs(radius - RADIUS) <= radial_limit) & (arc_distance <= arc_limit)


def wendland_c2(distance: np.ndarray, support: float = KERNEL_SUPPORT) -> np.ndarray:
    q = np.asarray(distance, dtype=float) / support
    positive = np.maximum(1.0 - q, 0.0)
    return positive**4 * (4.0 * q + 1.0)


def interpolate_periodic(grid: SpectralGrid, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coordinates = np.vstack((np.asarray(x) / grid.dx, np.asarray(y) / grid.dy))
    return map_coordinates(
        field, coordinates, order=1, mode="grid-wrap", prefilter=False
    )


# RETIRED LEGACY PATH -------------------------------------------------------
# This contact-level split clips noisy scores to [-1,1] and is retained only
# so that historical files can be inspected.  It is not reached by the
# publication driver or any publication CLI mode.  Current Experiment 4 uses
# ``_build_geometry_contacts`` and an uncensored Gaussian perturbation.
def _build_contacts(
    grid: SpectralGrid,
    seed: int,
    gap_width: float = TRUTH_GAP_WIDTH,
    gap_angle: float = TRUTH_GAP_ANGLE,
) -> dict[str, np.ndarray | float | int]:
    rng = np.random.default_rng(seed)

    ring_angles = np.linspace(-np.pi, np.pi, 260, endpoint=False)
    ring_radius = RADIUS + rng.normal(0.0, 0.55, ring_angles.size)
    ring_x = CENTER[0] + ring_radius * np.cos(ring_angles)
    ring_y = CENTER[1] + ring_radius * np.sin(ring_angles)
    keep_ring = ~_points_in_holdout(ring_x, ring_y)
    ring_x = ring_x[keep_ring]
    ring_y = ring_y[keep_ring]

    background_x: list[float] = []
    background_y: list[float] = []
    while len(background_x) < 320:
        candidate_x = rng.uniform(0.0, LENGTH, 480)
        candidate_y = rng.uniform(0.0, LENGTH, 480)
        keep = ~_points_in_holdout(candidate_x, candidate_y)
        background_x.extend(candidate_x[keep].tolist())
        background_y.extend(candidate_y[keep].tolist())
    background_x_array = np.asarray(background_x[:320])
    background_y_array = np.asarray(background_y[:320])

    sample_x = np.r_[ring_x, background_x_array]
    sample_y = np.r_[ring_y, background_y_array]
    exact = lesion_score_field(sample_x, sample_y, gap_width, -1.0, gap_angle)
    noisy = np.clip(exact + rng.normal(0.0, CONTACT_NOISE_SD, exact.size), -1.0, 1.0)

    permutation = rng.permutation(noisy.size)
    n_validation = max(1, int(round(VALIDATION_FRACTION * noisy.size)))
    validation_index = permutation[:n_validation]
    training_index = permutation[n_validation:]
    return {
        "train_x": sample_x[training_index],
        "train_y": sample_y[training_index],
        "train_score": noisy[training_index],
        "train_exact": exact[training_index],
        "validation_x": sample_x[validation_index],
        "validation_y": sample_y[validation_index],
        "validation_score": noisy[validation_index],
        "validation_exact": exact[validation_index],
        "noise_rms": float(np.sqrt(np.mean((noisy - exact) ** 2))),
        "n_ring_contacts": int(ring_x.size),
        "n_training": int(training_index.size),
        "n_validation": int(validation_index.size),
    }


def _build_geometry_contacts(
    grid: SpectralGrid,
    seed: int,
    truth: np.ndarray,
    blackout_angle: float,
) -> dict[str, np.ndarray | float | int]:
    """Construct one fixed spatial-block acquisition outside a missing sector.

    Unlike the original calibration design, this function does not make a
    random contact-level validation split.  Every retained contact is used for
    fitting and evaluation is confined to the dense zero-confidence block.
    The same seed at a given block rotation gives the same contact coordinates
    and Gaussian noise draw for all five geometries.
    """

    rng = np.random.default_rng(seed)
    ring_angles = np.linspace(-np.pi, np.pi, 260, endpoint=False)
    ring_radius = RADIUS + rng.normal(0.0, 0.55, ring_angles.size)
    ring_x = CENTER[0] + ring_radius * np.cos(ring_angles)
    ring_y = CENTER[1] + ring_radius * np.sin(ring_angles)
    keep_ring = ~_points_in_holdout(
        ring_x, ring_y, blackout_width=BLACKOUT_ARC_WIDTH, blackout_angle=blackout_angle
    )
    ring_x = ring_x[keep_ring]
    ring_y = ring_y[keep_ring]

    background_x: list[float] = []
    background_y: list[float] = []
    while len(background_x) < 320:
        candidate_x = rng.uniform(0.0, LENGTH, 480)
        candidate_y = rng.uniform(0.0, LENGTH, 480)
        keep = ~_points_in_holdout(
            candidate_x,
            candidate_y,
            blackout_width=BLACKOUT_ARC_WIDTH,
            blackout_angle=blackout_angle,
        )
        background_x.extend(candidate_x[keep].tolist())
        background_y.extend(candidate_y[keep].tolist())

    sample_x = np.r_[ring_x, np.asarray(background_x[:320])]
    sample_y = np.r_[ring_y, np.asarray(background_y[:320])]
    exact = interpolate_periodic(grid, truth, sample_x, sample_y)
    noise_draw = rng.normal(0.0, CONTACT_NOISE_SD, exact.size)
    # Keep the prescribed Gaussian perturbation uncensored.  Clipping here
    # would replace the stated noise model by a boundary-censored law on the
    # many contacts whose exact score is close to +/-1, and would hide that
    # change from the reconstruction and calibration diagnostics.
    noisy = exact + noise_draw
    coordinates = np.ascontiguousarray(
        np.column_stack((sample_x, sample_y)), dtype="<f8"
    )
    noise_bytes = np.ascontiguousarray(noise_draw, dtype="<f8")
    return {
        "train_x": sample_x,
        "train_y": sample_y,
        "train_score": noisy,
        "train_exact": exact,
        "noise_rms": float(np.sqrt(np.mean((noisy - exact) ** 2))),
        "n_ring_contacts": int(ring_x.size),
        "n_training": int(noisy.size),
        "acquisition_coordinate_sha256": hashlib.sha256(
            coordinates.tobytes()
        ).hexdigest(),
        "contact_noise_draw_sha256": hashlib.sha256(
            noise_bytes.tobytes()
        ).hexdigest(),
    }


def compact_observation_fields(
    grid: SpectralGrid,
    contacts: dict[str, np.ndarray | float | int],
    blackout_width: float = BLACKOUT_ARC_WIDTH,
    blackout_angle: float = BLACKOUT_ANGLE,
) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.zeros((grid.nx, grid.ny), dtype=float)
    data_forcing = np.zeros_like(confidence)
    train_x = np.asarray(contacts["train_x"])
    train_y = np.asarray(contacts["train_y"])
    train_score = np.asarray(contacts["train_score"])
    for x_i, y_i, score_i in zip(train_x, train_y, train_score):
        dx = np.minimum(np.abs(grid.X - x_i), LENGTH - np.abs(grid.X - x_i))
        dy = np.minimum(np.abs(grid.Y - y_i), LENGTH - np.abs(grid.Y - y_i))
        kernel = KERNEL_WEIGHT * wendland_c2(np.sqrt(dx * dx + dy * dy))
        confidence += kernel
        data_forcing += kernel * score_i

    hidden = holdout_mask(grid, blackout_width, blackout_angle)
    confidence[hidden] = 0.0
    data_forcing[hidden] = 0.0
    return confidence, data_forcing


def screened_initial_state(
    grid: SpectralGrid,
    confidence: np.ndarray,
    data_forcing: np.ndarray,
    epsilon: float = 0.05,
    length_scale: float = 1.50,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve (epsilon+lambda)u0-length_scale^2 L_K u0=f.

    The right-hand side uses observations only.  The positive epsilon fixes the
    constant mode and the elliptic term screens pointwise sampling noise.
    """
    shape = confidence.shape

    def matrix_vector(flat: np.ndarray) -> np.ndarray:
        value = flat.reshape(shape)
        result = (epsilon + confidence) * value - length_scale**2 * grid.anisotropic_laplacian(value)
        return result.ravel()

    diagonal_mean = epsilon + float(np.mean(confidence))

    def precondition(flat: np.ndarray) -> np.ndarray:
        value = flat.reshape(shape)
        transformed = grid.fft(value) / (diagonal_mean + length_scale**2 * grid.A)
        return grid.ifft(transformed).ravel()

    operator = LinearOperator((confidence.size, confidence.size), matvec=matrix_vector, dtype=float)
    preconditioner = LinearOperator(
        (confidence.size, confidence.size), matvec=precondition, dtype=float
    )
    iteration_count = 0

    def callback(_: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    solution, info = cg(
        operator,
        data_forcing.ravel(),
        M=preconditioner,
        rtol=1.0e-10,
        atol=1.0e-12,
        maxiter=800,
        callback=callback,
    )
    if info != 0:
        raise RuntimeError(f"screened initial-state solve failed with info={info}")
    state = solution.reshape(shape)
    residual = matrix_vector(solution) - data_forcing.ravel()
    relative_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(data_forcing.ravel()), 1.0)
    )
    return state, {
        "screen_iterations": float(iteration_count),
        "screen_relative_residual": relative_residual,
        "screen_min": float(np.min(state)),
        "screen_max": float(np.max(state)),
    }


def phase_parameters(method: str, dt: float = PHASE_DT) -> PhaseParameters:
    if method == "passive":
        return PhaseParameters(
            mu=0.30,
            nu=0.0,
            dt=dt,
            classifier="passive",
            admm_tol=5.0e-7,
            rho_factor=20.0,
        )
    if method != "graph":
        raise ValueError(f"unknown reconstruction method {method!r}")
    return PhaseParameters(
        mu=0.30,
        nu=0.20,
        dt=dt,
        classifier="graph",
        admm_tol=5.0e-7,
        admm_maxiter=8000,
        rho_factor=20.0,
    )


def run_reconstruction_history(
    grid: SpectralGrid,
    initial: np.ndarray,
    confidence: np.ndarray,
    data_forcing: np.ndarray,
    method: str,
    nsteps: int,
    dt: float = PHASE_DT,
) -> tuple[object, list[np.ndarray]]:
    state, history = solve_phase(
        grid,
        initial,
        np.zeros_like(initial),
        confidence,
        phase_parameters(method, dt),
        nsteps,
        keep_history=True,
        data_forcing=data_forcing,
    )
    if history is None:
        raise RuntimeError("phase history was not returned")
    return state, history


def terminal_average(history: list[np.ndarray], step: int, window: int = TERMINAL_WINDOW) -> np.ndarray:
    if not (window <= step < len(history)):
        raise ValueError("terminal averaging window is incompatible with the requested step")
    return np.mean(history[step - window + 1 : step + 1], axis=0)


def validation_error(
    grid: SpectralGrid,
    field: np.ndarray,
    contacts: dict[str, np.ndarray | float | int],
) -> float:
    predicted = interpolate_periodic(
        grid,
        field,
        np.asarray(contacts["validation_x"]),
        np.asarray(contacts["validation_y"]),
    )
    observed = np.asarray(contacts["validation_score"])
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def select_horizon(
    calibration_seeds: tuple[int, ...],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Select passive and graph horizons without consulting test truth.

    For each estimator, choose the shortest candidate whose mean held-out-contact
    RMSE is within one standard error of that estimator's smallest mean.  This
    favours the less evolved reconstruction when calibration errors are not
    distinguishable at the resolution of the calibration seeds.
    """
    grid = reconstruction_grid()
    maximum_step = int(round(max(CANDIDATE_HORIZONS) / PHASE_DT))
    rows: list[dict[str, float | int | str]] = []
    for method in ("passive", "graph"):
        for seed in calibration_seeds:
            contacts = _build_contacts(grid, seed)
            confidence, data_forcing = compact_observation_fields(grid, contacts)
            initial, screen = screened_initial_state(grid, confidence, data_forcing)
            state, history = run_reconstruction_history(
                grid, initial, confidence, data_forcing, method, maximum_step
            )
            for horizon in CANDIDATE_HORIZONS:
                step = int(round(horizon / PHASE_DT))
                reconstruction = terminal_average(history, step)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "horizon": horizon,
                        "validation_rmse": validation_error(
                            grid, reconstruction, contacts
                        ),
                        "n_training": int(contacts["n_training"]),
                        "n_validation": int(contacts["n_validation"]),
                        "reconstruction_n": grid.nx,
                        "reconstruction_dx_mm": grid.dx,
                        "kx_weight": grid.kx_weight,
                        "ky_weight": grid.ky_weight,
                        "kernel_support_mm": KERNEL_SUPPORT,
                        "kernel_weight": KERNEL_WEIGHT,
                        "blackout_arc_width_mm": BLACKOUT_ARC_WIDTH,
                        "blackout_angle_deg": np.rad2deg(BLACKOUT_ANGLE),
                        "blackout_buffer_mm": HOLDOUT_BUFFER,
                        "contact_noise_sd": CONTACT_NOISE_SD,
                        "zero_confidence_fraction": float(np.mean(confidence == 0.0)),
                        "holdout_confidence_max": float(
                            np.max(confidence[holdout_mask(grid)])
                        ),
                        "holdout_data_forcing_max_abs": float(
                            np.max(np.abs(data_forcing[holdout_mask(grid)]))
                        ),
                        "screen_relative_residual": screen["screen_relative_residual"],
                        "max_state_residual": max(
                            diagnostic["state_residual"]
                            for diagnostic in state.diagnostics[:step]
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["method", "horizon"], as_index=False)["validation_rmse"]
        .agg(["mean", "std", "count"])
    )
    summary["standard_error"] = (
        summary["std"] / np.sqrt(summary["count"])
    ).fillna(0.0)
    selected: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    for method in ("passive", "graph"):
        method_summary = summary.loc[summary["method"] == method].copy()
        minimum = method_summary.sort_values(["mean", "horizon"]).iloc[0]
        threshold = float(minimum["mean"] + minimum["standard_error"])
        eligible = method_summary.loc[method_summary["mean"] <= threshold]
        selected[method] = float(eligible["horizon"].min())
        thresholds[method] = threshold
    frame = frame.merge(summary, on=["method", "horizon"], how="left")
    frame["one_se_threshold"] = frame["method"].map(thresholds)
    frame["selected_horizon"] = frame["method"].map(selected)
    frame["selection_rule"] = "shortest mean RMSE within one SE of minimum"
    return selected, frame


def _boundary_metrics(prediction: np.ndarray, truth: np.ndarray, spacing: float) -> dict[str, float]:
    predicted_mask = prediction > 0.0
    truth_mask = truth > 0.0
    predicted_boundary = predicted_mask ^ binary_erosion(predicted_mask)
    truth_boundary = truth_mask ^ binary_erosion(truth_mask)
    if not (np.any(predicted_boundary) and np.any(truth_boundary)):
        return {"boundary_mean_mm": np.nan, "boundary_hd95_mm": np.nan}
    to_truth = distance_transform_edt(~truth_boundary, sampling=spacing)[predicted_boundary]
    to_prediction = distance_transform_edt(~predicted_boundary, sampling=spacing)[truth_boundary]
    distances = np.r_[to_truth, to_prediction]
    return {
        "boundary_mean_mm": float(np.mean(distances)),
        "boundary_hd95_mm": float(np.quantile(distances, 0.95)),
    }


def _ring_topology(grid: SpectralGrid, field: np.ndarray) -> dict[str, float | int]:
    """Threshold topology on a densely interpolated periodic ring centreline."""
    ntheta = 4096
    theta = np.linspace(-np.pi, np.pi, ntheta, endpoint=False)
    x = CENTER[0] + RADIUS * np.cos(theta)
    y = CENTER[1] + RADIUS * np.sin(theta)
    profile = interpolate_periodic(grid, field, x, y)
    viable = profile < 0.0
    if np.all(viable):
        components = 1
    elif not np.any(viable):
        components = 0
    else:
        components = int(np.sum(viable & ~np.roll(viable, 1)))
    return {
        "ring_viable_fraction": float(np.mean(viable)),
        "ring_viable_component_count": components,
        "complete_barrier_indicator": int(not np.any(viable)),
    }


def _ring_gap_geometry(grid: SpectralGrid, field: np.ndarray) -> dict[str, float | int]:
    """Measure every thresholded viable component on the ring centreline."""

    ntheta = 4096
    theta = np.linspace(-np.pi, np.pi, ntheta, endpoint=False)
    x = CENTER[0] + RADIUS * np.cos(theta)
    y = CENTER[1] + RADIUS * np.sin(theta)
    viable = interpolate_periodic(grid, field, x, y) < 0.0
    circumference = 2.0 * np.pi * RADIUS
    if not np.any(viable):
        count = 0
        largest = 0.0
    elif np.all(viable):
        count = 1
        largest = circumference
    else:
        starts = np.flatnonzero(viable & ~np.roll(viable, 1))
        lengths = []
        for start in starts:
            length = 0
            while length < ntheta and viable[(start + length) % ntheta]:
                length += 1
            lengths.append(length)
        count = len(lengths)
        largest = max(lengths) * circumference / ntheta
    return {
        "gap_count": int(count),
        "total_gap_width_mm": float(np.mean(viable) * circumference),
        "largest_gap_width_mm": float(largest),
    }


def _continuous_calibration_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Calibration diagnostics on the affine probability scale ``(u+1)/2``.

    No score is clipped.  Empty fixed-width bins make zero contribution to the
    expected calibration error.  The returned out-of-range fraction therefore
    exposes rather than conceals any violation of the nominal score range.
    """

    predicted = 0.5 * (np.asarray(prediction, dtype=float).ravel() + 1.0)
    observed = 0.5 * (np.asarray(truth, dtype=float).ravel() + 1.0)
    if len(predicted) == 0 or len(predicted) != len(observed):
        raise ValueError("calibration arrays must be non-empty and paired")
    design = np.column_stack((np.ones_like(predicted), predicted))
    intercept, slope = np.linalg.lstsq(design, observed, rcond=None)[0]
    if np.std(predicted) > 0.0 and np.std(observed) > 0.0:
        correlation = float(np.corrcoef(predicted, observed)[0, 1])
    else:
        correlation = np.nan
    bin_index = np.minimum(
        np.floor(np.clip(predicted, 0.0, 1.0 - np.finfo(float).eps) * n_bins).astype(int),
        n_bins - 1,
    )
    ece = 0.0
    for index in range(n_bins):
        selected = bin_index == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(predicted[selected]) - np.mean(observed[selected]))
            )
    return {
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "calibration_slope_abs_error": float(abs(slope - 1.0)),
        "calibration_in_the_large": float(np.mean(predicted) - np.mean(observed)),
        "calibration_ece": float(ece),
        "score_correlation": correlation,
        "scaled_score_mse": float(np.mean((predicted - observed) ** 2)),
        "out_of_range_fraction": float(
            np.mean((predicted < 0.0) | (predicted > 1.0))
        ),
    }


def _fixed_width_reliability_rows(
    prediction: np.ndarray,
    truth: np.ndarray,
    geometry: str,
    method: str,
    scope: str,
    n_bins: int = 10,
) -> list[dict[str, float | int | str]]:
    predicted = 0.5 * (np.asarray(prediction, dtype=float).ravel() + 1.0)
    observed = 0.5 * (np.asarray(truth, dtype=float).ravel() + 1.0)
    bin_index = np.minimum(
        np.floor(np.clip(predicted, 0.0, 1.0 - np.finfo(float).eps) * n_bins).astype(int),
        n_bins - 1,
    )
    rows: list[dict[str, float | int | str]] = []
    for index in range(n_bins):
        selected = bin_index == index
        rows.append(
            {
                "scope": scope,
                "geometry": geometry,
                "method": method,
                "bin": index + 1,
                "bin_lower": index / n_bins,
                "bin_upper": (index + 1) / n_bins,
                "n": int(np.sum(selected)),
                "mean_predicted": (
                    float(np.mean(predicted[selected])) if np.any(selected) else np.nan
                ),
                "mean_truth": (
                    float(np.mean(observed[selected])) if np.any(selected) else np.nan
                ),
            }
        )
    return rows


def _gap_region(grid: SpectralGrid) -> np.ndarray:
    _, _, radius, arc_distance = lesion_components(
        grid.X, grid.Y, TRUTH_GAP_WIDTH, TRUTH_GAP_ANGLE, TRANSITION
    )
    return (
        (np.abs(radius - RADIUS) <= 0.5 * LESION_WIDTH + TRANSITION)
        & (arc_distance <= 0.5 * TRUTH_GAP_WIDTH + TRANSITION)
    )


def _field_subcell_average(
    grid: SpectralGrid,
    field: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    subcells: int = 2,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=float)
    base_i, base_j = np.meshgrid(
        np.arange(grid.nx, dtype=float),
        np.arange(grid.ny, dtype=float),
        indexing="ij",
    )
    for a in range(subcells):
        for b in range(subcells):
            coordinates = np.vstack(
                (
                    (base_i + (a + 0.5) / subcells).ravel(),
                    (base_j + (b + 0.5) / subcells).ravel(),
                )
            )
            sampled = map_coordinates(
                field,
                coordinates,
                order=1,
                mode="grid-wrap",
                prefilter=False,
            ).reshape(field.shape)
            result += transform(sampled)
    return result / float(subcells * subcells)


def score_to_subcell_diffusivity(grid: SpectralGrid, score: np.ndarray, subcells: int = 2) -> np.ndarray:
    return _field_subcell_average(
        grid,
        score,
        lambda value: diffusivity_factor(value, ETA_MIN, ETA_SLOPE),
        subcells,
    )


def score_to_fv_diffusivity(
    source_grid: SpectralGrid,
    score: np.ndarray,
    n_fv: int = PRODUCTION_EP_N,
    subcells: int = 3,
) -> np.ndarray:
    """Periodically evaluate a spectral score on FV subcells, then map to eta."""
    spacing = LENGTH / n_fv
    result = np.zeros((n_fv, n_fv), dtype=float)
    for a in range(subcells):
        x = (np.arange(n_fv) + (a + 0.5) / subcells) * spacing
        source_i = x / source_grid.dx
        for b in range(subcells):
            y = (np.arange(n_fv) + (b + 0.5) / subcells) * spacing
            source_j = y / source_grid.dy
            coordinates = np.vstack(
                (
                    np.repeat(source_i, n_fv),
                    np.tile(source_j, n_fv),
                )
            )
            sampled = map_coordinates(
                score,
                coordinates,
                order=1,
                mode="grid-wrap",
                prefilter=False,
            ).reshape((n_fv, n_fv))
            result += diffusivity_factor(sampled, ETA_MIN, ETA_SLOPE)
    return result / float(subcells * subcells)


def analytic_subcell_diffusivity(
    n: int,
    gap_width: float,
    gap_diffusivity: float,
    gap_angle: float,
    subcells: int = 3,
) -> np.ndarray:
    spacing = LENGTH / n
    result = np.zeros((n, n), dtype=float)
    for a in range(subcells):
        x = (np.arange(n) + (a + 0.5) / subcells) * spacing
        for b in range(subcells):
            y = (np.arange(n) + (b + 0.5) / subcells) * spacing
            X, Y = np.meshgrid(x, y, indexing="ij")
            result += lesion_diffusivity_field(X, Y, gap_width, gap_diffusivity, gap_angle)
    return result / float(subcells * subcells)


def realised_centerline_gap_diffusivity(
    gap_width: float,
    gap_diffusivity: float,
    gap_angle: float,
) -> float:
    """Value of the smooth analytic profile at the gap centre on the ring."""
    x = np.asarray([[CENTER[0] + RADIUS * np.cos(gap_angle)]])
    y = np.asarray([[CENTER[1] + RADIUS * np.sin(gap_angle)]])
    return float(
        lesion_diffusivity_field(
            x, y, gap_width, gap_diffusivity, gap_angle
        )[0, 0]
    )


def ep_parameters(
    n: int,
    dt: float = PRODUCTION_EP_DT,
    face_average: str = "harmonic",
) -> EPParameters:
    spacing = LENGTH / n
    return EPParameters(
        dx=spacing,
        dy=spacing,
        dt=dt,
        d_long=0.32,
        d_trans=0.0512,
        eta_min=ETA_MIN,
        conductivity_slope=ETA_SLOPE,
        face_average=face_average,
    )


@lru_cache(maxsize=None)
def viable_capacity(
    n: int,
    fiber_axis: str = "x",
    face_average: str = "harmonic",
) -> float:
    scale = np.ones((n, n), dtype=float)
    result = annular_capacity(
        scale,
        ep_parameters(n, face_average=face_average),
        CENTER,
        12.0,
        18.0,
        fiber_axis,
    )
    return result.value


def capacity_metrics(
    diffusivity: np.ndarray,
    fiber_axis: str = "x",
    face_average: str = "harmonic",
) -> tuple[dict[str, float], np.ndarray]:
    n = diffusivity.shape[0]
    result = annular_capacity(
        diffusivity,
        ep_parameters(n, face_average=face_average),
        CENTER,
        inner_radius=12.0,
        outer_radius=18.0,
        fiber_axis=fiber_axis,
    )
    values = {
        "capacity": result.value,
        "normalized_capacity": result.value
        / viable_capacity(n, fiber_axis, face_average),
        "capacity_inner_flux": result.inner_flux,
        "capacity_outer_flux": result.outer_flux,
        "capacity_energy": result.energy,
        "capacity_relative_residual": result.relative_residual,
        "capacity_flux_energy_defect": result.flux_energy_defect,
    }
    return values, result.potential


def _diagnostic_maxima(
    state: object | None,
    steps: int | None = None,
) -> dict[str, float]:
    names = {
        "max_state_residual": "state_residual",
        "max_primal_residual": "primal_residual",
        "max_dual_residual": "dual_residual",
        "max_box_residual": "box_residual",
        "max_complementarity_residual": "complementarity_residual",
        "max_graph_projection_residual": "graph_projection_residual",
        "max_mass_defect": "mass_defect",
        "max_laplacian_tail_energy": "tail_laplacian",
        "max_iterations": "admm_iterations",
    }
    if state is None:
        return {name: np.nan for name in names}
    diagnostics = state.diagnostics if steps is None else state.diagnostics[:steps]
    if not diagnostics:
        raise ValueError("at least one diagnostic step is required")
    return {
        output: float(max(item[source] for item in diagnostics))
        for output, source in names.items()
    }


def run_sparse_reconstruction(
    selected_horizons: dict[str, float],
    test_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    grid = reconstruction_grid()
    methods = ("screened", "passive", "graph")
    truth = lesion_score_field(grid.X, grid.Y)
    truth_diffusivity = score_to_fv_diffusivity(grid, truth)
    truth_capacity, truth_potential = capacity_metrics(truth_diffusivity)
    truth_topology = _ring_topology(grid, truth)
    local_gap = _gap_region(grid)

    rows: list[dict[str, float | int | str]] = []
    fields_by_seed: dict[int, dict[str, np.ndarray]] = {}
    representative: dict[str, np.ndarray] = {
        "truth": truth,
        "truth_diffusivity": truth_diffusivity,
        "truth_capacity_potential": truth_potential,
    }

    for seed in test_seeds:
        contacts = _build_contacts(grid, seed)
        confidence, data_forcing = compact_observation_fields(grid, contacts)
        initial, screen = screened_initial_state(grid, confidence, data_forcing)
        fields: dict[str, np.ndarray] = {"screened": initial}
        states: dict[str, object | None] = {"screened": None}
        for method in ("passive", "graph"):
            selected_step = int(round(selected_horizons[method] / PHASE_DT))
            state, history = run_reconstruction_history(
                grid,
                initial,
                confidence,
                data_forcing,
                method,
                selected_step,
            )
            fields[method] = terminal_average(history, selected_step)
            states[method] = state

        fields_by_seed[seed] = fields
        truth_gap = ring_gap_width(
            truth, grid.X, grid.Y, CENTER, RADIUS, gap_angle=TRUTH_GAP_ANGLE
        )
        for method in methods:
            field = fields[method]
            metrics = segmentation_metrics(field, truth)
            boundary = _boundary_metrics(field, truth, grid.dx)
            topology = _ring_topology(grid, field)
            gap_width = ring_gap_width(
                field, grid.X, grid.Y, CENTER, RADIUS, gap_angle=TRUTH_GAP_ANGLE
            )
            diffusivity = score_to_fv_diffusivity(grid, field)
            capacity, _ = capacity_metrics(diffusivity)
            row = {
                "seed": seed,
                "method": method,
                "selected_horizon": selected_horizons.get(method, 0.0),
                **metrics,
                **boundary,
                **topology,
                "truth_ring_viable_component_count": truth_topology[
                    "ring_viable_component_count"
                ],
                "ring_topology_correct": int(
                    topology["ring_viable_component_count"]
                    == truth_topology["ring_viable_component_count"]
                ),
                "gap_width_mm": gap_width,
                "truth_zero_level_gap_width_mm": truth_gap,
                "gap_width_abs_error_mm": abs(gap_width - truth_gap),
                "gap_region_rmse": float(np.sqrt(np.mean((field[local_gap] - truth[local_gap]) ** 2))),
                "normalized_capacity": capacity["normalized_capacity"],
                "truth_normalized_capacity": truth_capacity["normalized_capacity"],
                "capacity_abs_error": abs(
                    capacity["normalized_capacity"] - truth_capacity["normalized_capacity"]
                ),
                "capacity": capacity["capacity"],
                "capacity_inner_flux": capacity["capacity_inner_flux"],
                "capacity_outer_flux": capacity["capacity_outer_flux"],
                "capacity_energy": capacity["capacity_energy"],
                "capacity_relative_residual": capacity["capacity_relative_residual"],
                "capacity_flux_energy_defect": capacity["capacity_flux_energy_defect"],
                "fv_n": PRODUCTION_EP_N,
                "fv_dx_mm": LENGTH / PRODUCTION_EP_N,
                "subcells_per_axis": SUBCELLS_PER_AXIS,
                "validation_rmse": validation_error(grid, field, contacts),
                "n_training": int(contacts["n_training"]),
                "n_validation": int(contacts["n_validation"]),
                "noise_rms": float(contacts["noise_rms"]),
                "contact_noise_sd": CONTACT_NOISE_SD,
                "reconstruction_n": grid.nx,
                "reconstruction_dx_mm": grid.dx,
                "kx_weight": grid.kx_weight,
                "ky_weight": grid.ky_weight,
                "kernel_support_mm": KERNEL_SUPPORT,
                "kernel_weight": KERNEL_WEIGHT,
                "blackout_arc_width_mm": BLACKOUT_ARC_WIDTH,
                "blackout_angle_deg": np.rad2deg(BLACKOUT_ANGLE),
                "blackout_buffer_mm": HOLDOUT_BUFFER,
                "zero_confidence_fraction": float(np.mean(confidence == 0.0)),
                "holdout_confidence_max": float(np.max(confidence[holdout_mask(grid)])),
                "holdout_data_forcing_max_abs": float(
                    np.max(np.abs(data_forcing[holdout_mask(grid)]))
                ),
                **screen,
                **_diagnostic_maxima(states[method]),
            }
            rows.append(row)

        if seed == REPRESENTATIVE_SEED:
            representative.update(fields)
            representative.update(
                {
                    "confidence": confidence,
                    "data_forcing": data_forcing,
                    "holdout_mask": holdout_mask(grid).astype(float),
                    "train_x": np.asarray(contacts["train_x"]),
                    "train_y": np.asarray(contacts["train_y"]),
                    "train_score": np.asarray(contacts["train_score"]),
                    "validation_x": np.asarray(contacts["validation_x"]),
                    "validation_y": np.asarray(contacts["validation_y"]),
                }
            )

    ensemble = pd.DataFrame(rows)

    # One representative sensitivity study: T/2, T, 2T and screened versus
    # constant initialisation.  Pseudo-time is not presented as convergence to
    # a steady state.
    contacts = _build_contacts(grid, REPRESENTATIVE_SEED)
    confidence, data_forcing = compact_observation_fields(grid, contacts)
    screened, _ = screened_initial_state(grid, confidence, data_forcing)
    graph_horizon = selected_horizons["graph"]
    maximum_horizon = 2.0 * graph_horizon
    maximum_step = int(round(maximum_horizon / PHASE_DT))
    sensitivity_rows: list[dict[str, float | str]] = []
    for initial_label, initial in (
        ("screened", screened),
        ("constant_-0.8", np.full_like(screened, -0.8)),
    ):
        state, history = run_reconstruction_history(
            grid,
            initial,
            confidence,
            data_forcing,
            "graph",
            maximum_step,
        )
        for multiplier in (0.5, 1.0, 2.0):
            horizon = multiplier * graph_horizon
            step = int(round(horizon / PHASE_DT))
            field = terminal_average(history, step)
            metrics = segmentation_metrics(field, truth)
            diffusivity = score_to_fv_diffusivity(grid, field)
            capacity, _ = capacity_metrics(diffusivity)
            sensitivity_rows.append(
                {
                    "initialisation": initial_label,
                    "horizon_multiplier": multiplier,
                    "horizon": horizon,
                    "rmse": metrics["rmse"],
                    "dice": metrics["dice"],
                    "validation_rmse": validation_error(grid, field, contacts),
                    "normalized_capacity": capacity["normalized_capacity"],
                    "capacity_abs_error": abs(
                        capacity["normalized_capacity"] - truth_capacity["normalized_capacity"]
                    ),
                    "last_step_rms_change": float(
                        np.sqrt(np.mean((history[step] - history[step - 1]) ** 2))
                    ),
                    "max_state_residual": max(
                        item["state_residual"] for item in state.diagnostics[:step]
                    ),
                }
            )
    return ensemble, pd.DataFrame(sensitivity_rows), fields_by_seed, representative


def run_common_horizon_ablation(
    test_seeds: tuple[int, ...] = TEST_SEEDS,
    horizons: tuple[float, ...] = COMMON_HORIZONS,
) -> pd.DataFrame:
    """Compare passive and graph reconstructions at identical pseudo-times.

    Each method is advanced once per seed to the largest requested horizon. Earlier
    terminal averages are read from that trajectory, so observations, screened initial
    state, grid, step size, terminal window, seed, and horizon are paired exactly.
    """

    if len(test_seeds) != len(set(test_seeds)) or not test_seeds:
        raise ValueError("common-horizon ablation needs unique non-empty test seeds")
    unknown = sorted(set(test_seeds).difference(TEST_SEEDS))
    if unknown:
        raise ValueError(f"unknown common-horizon test seeds: {unknown}")
    horizon_values = tuple(sorted(float(value) for value in horizons))
    if (
        not horizon_values
        or len(horizon_values) != len(set(horizon_values))
        or any(not np.isfinite(value) or value <= 0.0 for value in horizon_values)
    ):
        raise ValueError("common horizons must be distinct finite positive values")
    steps_by_horizon = {
        horizon: int(round(horizon / PHASE_DT)) for horizon in horizon_values
    }
    for horizon, step in steps_by_horizon.items():
        if not np.isclose(step * PHASE_DT, horizon, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"common horizon {horizon:g} is not a multiple of pseudo-time step")
        if step < TERMINAL_WINDOW:
            raise ValueError(f"common horizon {horizon:g} is shorter than the terminal window")

    grid = reconstruction_grid()
    truth = lesion_score_field(grid.X, grid.Y)
    truth_diffusivity = score_to_fv_diffusivity(grid, truth)
    truth_capacity, _ = capacity_metrics(truth_diffusivity)
    truth_topology = _ring_topology(grid, truth)
    truth_gap = ring_gap_width(
        truth, grid.X, grid.Y, CENTER, RADIUS, gap_angle=TRUTH_GAP_ANGLE
    )
    local_gap = _gap_region(grid)
    maximum_step = max(steps_by_horizon.values())
    rows: list[dict[str, float | int | str]] = []

    for seed in test_seeds:
        contacts = _build_contacts(grid, seed)
        confidence, data_forcing = compact_observation_fields(grid, contacts)
        initial, screen = screened_initial_state(grid, confidence, data_forcing)
        for method in ("passive", "graph"):
            state, history = run_reconstruction_history(
                grid,
                initial,
                confidence,
                data_forcing,
                method,
                maximum_step,
            )
            for horizon in horizon_values:
                step = steps_by_horizon[horizon]
                field = terminal_average(history, step)
                metrics = segmentation_metrics(field, truth)
                boundary = _boundary_metrics(field, truth, grid.dx)
                topology = _ring_topology(grid, field)
                gap_width = ring_gap_width(
                    field,
                    grid.X,
                    grid.Y,
                    CENTER,
                    RADIUS,
                    gap_angle=TRUTH_GAP_ANGLE,
                )
                diffusivity = score_to_fv_diffusivity(grid, field)
                capacity, _ = capacity_metrics(diffusivity)
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "common_horizon": horizon,
                        "is_method_selected_horizon": int(
                            (method == "passive" and np.isclose(horizon, 0.60))
                            or (method == "graph" and np.isclose(horizon, 1.20))
                        ),
                        "comparison_design": "same_seed_same_horizon",
                        "pseudo_dt": PHASE_DT,
                        "terminal_window_states": TERMINAL_WINDOW,
                        "terminal_window_pseudo_time": TERMINAL_WINDOW * PHASE_DT,
                        **metrics,
                        **boundary,
                        **topology,
                        "truth_ring_viable_component_count": truth_topology[
                            "ring_viable_component_count"
                        ],
                        "ring_topology_correct": int(
                            topology["ring_viable_component_count"]
                            == truth_topology["ring_viable_component_count"]
                        ),
                        "gap_width_mm": gap_width,
                        "truth_zero_level_gap_width_mm": truth_gap,
                        "gap_width_abs_error_mm": abs(gap_width - truth_gap),
                        "gap_region_rmse": float(
                            np.sqrt(np.mean((field[local_gap] - truth[local_gap]) ** 2))
                        ),
                        "normalized_capacity": capacity["normalized_capacity"],
                        "truth_normalized_capacity": truth_capacity[
                            "normalized_capacity"
                        ],
                        "capacity_abs_error": abs(
                            capacity["normalized_capacity"]
                            - truth_capacity["normalized_capacity"]
                        ),
                        "capacity": capacity["capacity"],
                        "capacity_inner_flux": capacity["capacity_inner_flux"],
                        "capacity_outer_flux": capacity["capacity_outer_flux"],
                        "capacity_energy": capacity["capacity_energy"],
                        "capacity_relative_residual": capacity[
                            "capacity_relative_residual"
                        ],
                        "capacity_flux_energy_defect": capacity[
                            "capacity_flux_energy_defect"
                        ],
                        "fv_n": PRODUCTION_EP_N,
                        "fv_dx_mm": LENGTH / PRODUCTION_EP_N,
                        "subcells_per_axis": SUBCELLS_PER_AXIS,
                        "validation_rmse": validation_error(grid, field, contacts),
                        "n_training": int(contacts["n_training"]),
                        "n_validation": int(contacts["n_validation"]),
                        "noise_rms": float(contacts["noise_rms"]),
                        "contact_noise_sd": CONTACT_NOISE_SD,
                        "reconstruction_n": grid.nx,
                        "reconstruction_dx_mm": grid.dx,
                        "kx_weight": grid.kx_weight,
                        "ky_weight": grid.ky_weight,
                        "kernel_support_mm": KERNEL_SUPPORT,
                        "kernel_weight": KERNEL_WEIGHT,
                        "blackout_arc_width_mm": BLACKOUT_ARC_WIDTH,
                        "blackout_angle_deg": np.rad2deg(BLACKOUT_ANGLE),
                        "blackout_buffer_mm": HOLDOUT_BUFFER,
                        "zero_confidence_fraction": float(np.mean(confidence == 0.0)),
                        "holdout_confidence_max": float(
                            np.max(confidence[holdout_mask(grid)])
                        ),
                        "holdout_data_forcing_max_abs": float(
                            np.max(np.abs(data_forcing[holdout_mask(grid)]))
                        ),
                        **screen,
                        **_diagnostic_maxima(state, steps=step),
                    }
                )

    frame = pd.DataFrame(rows).sort_values(
        ["common_horizon", "seed", "method"], kind="mergesort"
    ).reset_index(drop=True)
    paired_metrics = (
        "rmse",
        "dice",
        "boundary_hd95_mm",
        "gap_width_abs_error_mm",
        "gap_region_rmse",
        "normalized_capacity",
        "capacity_abs_error",
        "validation_rmse",
    )
    pair_index = pd.MultiIndex.from_frame(frame[["seed", "common_horizon"]])
    for metric in paired_metrics:
        pivot = frame.pivot(
            index=["seed", "common_horizon"], columns="method", values=metric
        )
        if set(pivot.columns) != {"passive", "graph"} or pivot.isna().any().any():
            raise RuntimeError(f"common-horizon pairing failed for {metric}")
        difference = pivot["graph"] - pivot["passive"]
        frame[f"paired_graph_minus_passive_{metric}"] = difference.reindex(
            pair_index
        ).to_numpy(dtype=float)
    return frame


def summarize_common_horizon_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    """Return reproducible paired summaries for the fixed-horizon comparison."""

    required = {"seed", "method", "common_horizon", *COMMON_HORIZON_SUMMARY_METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"common-horizon table is missing summary fields: {sorted(missing)}")
    rows: list[dict[str, float | int | str]] = []
    for horizon in sorted(frame["common_horizon"].unique()):
        subset = frame.loc[np.isclose(frame["common_horizon"], horizon)]
        for metric in COMMON_HORIZON_SUMMARY_METRICS:
            pivot = subset.pivot(index="seed", columns="method", values=metric)
            if set(pivot.columns) != {"passive", "graph"} or pivot.isna().any().any():
                raise RuntimeError(
                    f"common-horizon summary pairing failed for T={horizon:g}, {metric}"
                )
            difference = (pivot["graph"] - pivot["passive"]).to_numpy(dtype=float)
            rng = np.random.default_rng(COMMON_HORIZON_BOOTSTRAP_SEED)
            bootstrap_means = np.mean(
                rng.choice(
                    difference,
                    size=(COMMON_HORIZON_BOOTSTRAP_RESAMPLES, len(difference)),
                    replace=True,
                ),
                axis=1,
            )
            lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
            rows.append(
                {
                    "common_horizon": float(horizon),
                    "metric": metric,
                    "n": len(difference),
                    "passive_mean": float(pivot["passive"].mean()),
                    "graph_mean": float(pivot["graph"].mean()),
                    "graph_minus_passive_mean": float(np.mean(difference)),
                    "graph_minus_passive_sample_sd": float(
                        np.std(difference, ddof=1)
                    ),
                    "interval_type": "paired_bootstrap_percentile_mean",
                    "lower_95": float(lower),
                    "upper_95": float(upper),
                    "bootstrap_resamples": COMMON_HORIZON_BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": COMMON_HORIZON_BOOTSTRAP_SEED,
                    "graph_lower_count": int(np.sum(difference < 0.0)),
                    "tied_count": int(np.sum(difference == 0.0)),
                    "graph_higher_count": int(np.sum(difference > 0.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["common_horizon", "metric"], kind="mergesort"
    ).reset_index(drop=True)


def refresh_common_horizon_ablation() -> None:
    """Recompute only the 16-seed common-horizon reconstruction comparison."""

    frame = run_common_horizon_ablation()
    path = DATA / "sparse_reconstruction_common_horizon.csv"
    frame.to_csv(path, index=False)
    summary = summarize_common_horizon_ablation(frame)
    summary.to_csv(DATA / "sparse_reconstruction_common_horizon_summary.csv", index=False)
    print(
        "common-horizon ablation written\n",
        frame.groupby(["common_horizon", "method"])[
            ["rmse", "dice", "capacity_abs_error", "gap_width_abs_error_mm"]
        ].mean(),
    )
    print(
        "run_metadata.json must be refreshed after all targeted outputs with "
        "`python code/run_applied.py --refresh-run-metadata`"
    )


def run_geometry_replicated_reconstruction(
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Run the fixed-budget, spatial-block reconstruction stress test.

    The five lesion geometries, four block rotations, acquisition seeds and
    common 120-step horizon are constants above.  Blocks are averaged within a
    geometry before any passive--graph contrast is formed; hence geometry, not
    a noise draw or an individual blackout, is the replication unit.
    """

    grid = reconstruction_grid()
    specifications = GEOMETRY_SPECS[:2] if quick else GEOMETRY_SPECS
    blackout_degrees = GEOMETRY_BLACKOUT_ANGLES_DEG[:2] if quick else GEOMETRY_BLACKOUT_ANGLES_DEG
    acquisition_seeds = GEOMETRY_ACQUISITION_SEEDS[:2] if quick else GEOMETRY_ACQUISITION_SEEDS
    if len(blackout_degrees) != len(acquisition_seeds):
        raise RuntimeError("geometry blackout rotations and acquisition seeds are unpaired")
    step = int(round(GEOMETRY_RECONSTRUCTION_HORIZON / PHASE_DT))
    if not np.isclose(step * PHASE_DT, GEOMETRY_RECONSTRUCTION_HORIZON):
        raise RuntimeError("geometry reconstruction budget is not a whole number of steps")

    block_rows: list[dict[str, float | int | str]] = []
    calibration_arrays: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    maps: dict[str, np.ndarray] = {}
    for specification in specifications:
        geometry = str(specification["geometry"])
        gaps = specification["gaps"]
        if not isinstance(gaps, tuple):
            raise TypeError("geometry gaps must be a tuple")
        truth = geometry_score_field(grid.X, grid.Y, gaps)
        truth_sha256 = hashlib.sha256(
            np.ascontiguousarray(truth, dtype="<f8").tobytes()
        ).hexdigest()
        truth_ring = _ring_gap_geometry(grid, truth)
        truth_diffusivity = score_to_fv_diffusivity(grid, truth)
        truth_capacity, _ = capacity_metrics(truth_diffusivity)
        maps[f"truth_{geometry}"] = truth

        for block_index, (angle_deg, seed) in enumerate(
            zip(blackout_degrees, acquisition_seeds), start=1
        ):
            angle = np.deg2rad(angle_deg)
            evaluation = holdout_mask(
                grid,
                blackout_width=BLACKOUT_ARC_WIDTH,
                blackout_angle=angle,
            )
            contacts = _build_geometry_contacts(grid, seed, truth, angle)
            confidence, data_forcing = compact_observation_fields(
                grid,
                contacts,
                blackout_width=BLACKOUT_ARC_WIDTH,
                blackout_angle=angle,
            )
            initial, screen = screened_initial_state(grid, confidence, data_forcing)
            fields: dict[str, np.ndarray] = {"screened": initial}
            states: dict[str, object | None] = {"screened": None}
            for method in ("passive", "graph"):
                state, history = run_reconstruction_history(
                    grid,
                    initial,
                    confidence,
                    data_forcing,
                    method,
                    step,
                )
                fields[method] = terminal_average(history, step)
                states[method] = state

            if (
                geometry == "wide_gap" and np.isclose(angle_deg, 90.0)
            ) or ("representative_truth" not in maps and block_index == 1):
                maps.update(
                    {
                        "representative_truth": truth,
                        "representative_confidence": confidence,
                        "representative_holdout_mask": evaluation.astype(float),
                        "representative_train_x": np.asarray(contacts["train_x"]),
                        "representative_train_y": np.asarray(contacts["train_y"]),
                        "representative_screened": fields["screened"],
                        "representative_passive": fields["passive"],
                        "representative_graph": fields["graph"],
                    }
                )

            truth_masked = truth[evaluation]
            truth_binary = truth_masked > 0.0
            for method, field in fields.items():
                predicted_masked = field[evaluation]
                predicted_binary = predicted_masked > 0.0
                true_positive = int(np.sum(predicted_binary & truth_binary))
                false_negative = int(np.sum(~predicted_binary & truth_binary))
                true_negative = int(np.sum(~predicted_binary & ~truth_binary))
                false_positive = int(np.sum(predicted_binary & ~truth_binary))
                sensitivity = true_positive / max(true_positive + false_negative, 1)
                specificity = true_negative / max(true_negative + false_positive, 1)
                dice_denominator = int(np.sum(predicted_binary) + np.sum(truth_binary))
                masked_dice = (
                    2.0 * true_positive / dice_denominator
                    if dice_denominator > 0
                    else 1.0
                )
                calibration = _continuous_calibration_metrics(
                    predicted_masked, truth_masked
                )
                ring = _ring_gap_geometry(grid, field)
                diffusivity = score_to_fv_diffusivity(grid, field)
                capacity, _ = capacity_metrics(diffusivity)
                diagnostics = _diagnostic_maxima(states[method], steps=step)
                block_rows.append(
                    {
                        "geometry": geometry,
                        "geometry_description": str(specification["description"]),
                        "block_index": block_index,
                        "blackout_angle_deg": angle_deg,
                        "acquisition_seed": seed,
                        "acquisition_coordinate_sha256": str(
                            contacts["acquisition_coordinate_sha256"]
                        ),
                        "contact_noise_draw_sha256": str(
                            contacts["contact_noise_draw_sha256"]
                        ),
                        "truth_sha256": truth_sha256,
                        "method": method,
                        "replication_unit": "geometry",
                        "within_geometry_repeat": "fixed_spatial_block",
                        "horizon_rule": "fixed_common_120_step_budget_no_selection",
                        "horizon": GEOMETRY_RECONSTRUCTION_HORIZON if method != "screened" else 0.0,
                        "pseudo_dt": PHASE_DT,
                        "terminal_window_states": TERMINAL_WINDOW if method != "screened" else 0,
                        "n_training": int(contacts["n_training"]),
                        "n_masked_grid_points": int(np.sum(evaluation)),
                        "masked_score_rmse": float(
                            np.sqrt(np.mean((predicted_masked - truth_masked) ** 2))
                        ),
                        "masked_score_mae": float(
                            np.mean(np.abs(predicted_masked - truth_masked))
                        ),
                        "masked_score_bias": float(
                            np.mean(predicted_masked - truth_masked)
                        ),
                        "masked_scaled_score_mse": calibration["scaled_score_mse"],
                        "masked_calibration_intercept": calibration["calibration_intercept"],
                        "masked_calibration_slope": calibration["calibration_slope"],
                        "masked_calibration_slope_abs_error": calibration[
                            "calibration_slope_abs_error"
                        ],
                        "masked_calibration_in_the_large": calibration[
                            "calibration_in_the_large"
                        ],
                        "masked_calibration_ece": calibration["calibration_ece"],
                        "masked_score_correlation": calibration["score_correlation"],
                        "masked_out_of_range_fraction": calibration[
                            "out_of_range_fraction"
                        ],
                        "masked_dice": float(masked_dice),
                        "masked_balanced_accuracy": float(
                            0.5 * (sensitivity + specificity)
                        ),
                        "masked_sensitivity": float(sensitivity),
                        "masked_specificity": float(specificity),
                        "whole_score_rmse": float(
                            np.sqrt(np.mean((field - truth) ** 2))
                        ),
                        "gap_count": ring["gap_count"],
                        "truth_gap_count": truth_ring["gap_count"],
                        "gap_count_abs_error": abs(
                            int(ring["gap_count"]) - int(truth_ring["gap_count"])
                        ),
                        "total_gap_width_mm": ring["total_gap_width_mm"],
                        "truth_total_gap_width_mm": truth_ring["total_gap_width_mm"],
                        "total_gap_width_abs_error_mm": abs(
                            float(ring["total_gap_width_mm"])
                            - float(truth_ring["total_gap_width_mm"])
                        ),
                        "largest_gap_width_mm": ring["largest_gap_width_mm"],
                        "truth_largest_gap_width_mm": truth_ring[
                            "largest_gap_width_mm"
                        ],
                        "largest_gap_width_abs_error_mm": abs(
                            float(ring["largest_gap_width_mm"])
                            - float(truth_ring["largest_gap_width_mm"])
                        ),
                        "normalized_capacity": capacity["normalized_capacity"],
                        "truth_normalized_capacity": truth_capacity[
                            "normalized_capacity"
                        ],
                        "capacity_abs_error": abs(
                            capacity["normalized_capacity"]
                            - truth_capacity["normalized_capacity"]
                        ),
                        "capacity": capacity["capacity"],
                        "capacity_inner_flux": capacity["capacity_inner_flux"],
                        "capacity_outer_flux": capacity["capacity_outer_flux"],
                        "capacity_energy": capacity["capacity_energy"],
                        "capacity_relative_residual": capacity[
                            "capacity_relative_residual"
                        ],
                        "capacity_flux_energy_defect": capacity[
                            "capacity_flux_energy_defect"
                        ],
                        "score_min": float(np.min(field)),
                        "score_max": float(np.max(field)),
                        "reconstruction_n": grid.nx,
                        "reconstruction_dx_mm": grid.dx,
                        "fv_n": PRODUCTION_EP_N,
                        "subcells_per_axis": SUBCELLS_PER_AXIS,
                        "blackout_arc_width_mm": BLACKOUT_ARC_WIDTH,
                        "blackout_buffer_mm": HOLDOUT_BUFFER,
                        "zero_confidence_fraction": float(np.mean(confidence == 0.0)),
                        "holdout_confidence_max": float(np.max(confidence[evaluation])),
                        "holdout_data_forcing_max_abs": float(
                            np.max(np.abs(data_forcing[evaluation]))
                        ),
                        "noise_rms": float(contacts["noise_rms"]),
                        "observation_noise_model": "uncensored_gaussian",
                        "observation_min": float(
                            np.min(np.asarray(contacts["train_score"]))
                        ),
                        "observation_max": float(
                            np.max(np.asarray(contacts["train_score"]))
                        ),
                        "observation_out_of_range_fraction": float(
                            np.mean(
                                (np.asarray(contacts["train_score"]) < -1.0)
                                | (np.asarray(contacts["train_score"]) > 1.0)
                            )
                        ),
                        "screen_relative_residual": screen["screen_relative_residual"],
                        **diagnostics,
                    }
                )
                stored = calibration_arrays.setdefault(
                    (geometry, method), {"prediction": [], "truth": []}
                )
                stored["prediction"].append(predicted_masked.copy())
                stored["truth"].append(truth_masked.copy())

    blocks = pd.DataFrame(block_rows).sort_values(
        ["geometry", "block_index", "method"], kind="mergesort"
    ).reset_index(drop=True)
    unit_rows: list[dict[str, float | int | str]] = []
    reliability_rows: list[dict[str, float | int | str]] = []
    pooled_by_method: dict[str, dict[str, list[np.ndarray]]] = {
        method: {"prediction": [], "truth": []}
        for method in ("screened", "passive", "graph")
    }
    averaged_columns = (
        "masked_score_rmse",
        "masked_score_mae",
        "masked_score_bias",
        "masked_scaled_score_mse",
        "masked_dice",
        "masked_balanced_accuracy",
        "whole_score_rmse",
        "capacity_abs_error",
        "gap_count_abs_error",
        "total_gap_width_abs_error_mm",
        "largest_gap_width_abs_error_mm",
    )
    for (geometry, method), arrays in calibration_arrays.items():
        predicted = np.concatenate(arrays["prediction"])
        truth = np.concatenate(arrays["truth"])
        calibration = _continuous_calibration_metrics(predicted, truth)
        subset = blocks.loc[
            (blocks["geometry"] == geometry) & (blocks["method"] == method)
        ]
        row: dict[str, float | int | str] = {
            "geometry": geometry,
            "geometry_description": str(subset["geometry_description"].iloc[0]),
            "method": method,
            "replication_unit": "geometry",
            "n_spatial_blocks_averaged": len(subset),
            "block_averaging_rule": "arithmetic_mean_before_method_contrast",
            "horizon_rule": "fixed_common_120_step_budget_no_selection",
            "horizon": float(subset["horizon"].iloc[0]),
        }
        for column in averaged_columns:
            row[column] = float(subset[column].mean())
        row["masked_score_rmse_within_block_sd"] = float(
            subset["masked_score_rmse"].std(ddof=1)
        )
        row.update(
            {
                "masked_calibration_intercept": calibration[
                    "calibration_intercept"
                ],
                "masked_calibration_slope": calibration["calibration_slope"],
                "masked_calibration_slope_abs_error": calibration[
                    "calibration_slope_abs_error"
                ],
                "masked_calibration_in_the_large": calibration[
                    "calibration_in_the_large"
                ],
                "masked_calibration_ece": calibration["calibration_ece"],
                "masked_score_correlation": calibration["score_correlation"],
                "masked_out_of_range_fraction": calibration[
                    "out_of_range_fraction"
                ],
            }
        )
        unit_rows.append(row)
        reliability_rows.extend(
            _fixed_width_reliability_rows(
                predicted, truth, geometry, method, scope="geometry"
            )
        )
        pooled_by_method[method]["prediction"].append(predicted)
        pooled_by_method[method]["truth"].append(truth)

    units = pd.DataFrame(unit_rows).sort_values(
        ["geometry", "method"], kind="mergesort"
    ).reset_index(drop=True)
    summary_rows: list[dict[str, float | int | str]] = []
    for metric in GEOMETRY_SUMMARY_METRICS:
        pivot = units.pivot(index="geometry", columns="method", values=metric)
        difference = (pivot["graph"] - pivot["passive"]).to_numpy(dtype=float)
        tie_tolerance = 1.0e-14
        summary_rows.append(
            {
                "metric": metric,
                "n_geometries": len(difference),
                "screened_mean": float(pivot["screened"].mean()),
                "passive_mean": float(pivot["passive"].mean()),
                "graph_mean": float(pivot["graph"].mean()),
                "graph_minus_passive_mean": float(np.mean(difference)),
                "graph_minus_passive_median": float(np.median(difference)),
                "graph_minus_passive_min": float(np.min(difference)),
                "graph_minus_passive_max": float(np.max(difference)),
                "graph_better_geometry_count": int(
                    np.sum(difference < -tie_tolerance)
                ),
                "tied_geometry_count": int(
                    np.sum(np.abs(difference) <= tie_tolerance)
                ),
                "graph_worse_geometry_count": int(
                    np.sum(difference > tie_tolerance)
                ),
                "interval_type": "none_fixed_prespecified_scenario_set",
            }
        )
    summary = pd.DataFrame(summary_rows)

    for method, arrays in pooled_by_method.items():
        prediction = np.concatenate(arrays["prediction"])
        truth = np.concatenate(arrays["truth"])
        reliability_rows.extend(
            _fixed_width_reliability_rows(
                prediction,
                truth,
                "all_equal_size_geometries",
                method,
                scope="pooled_descriptive",
            )
        )
    reliability = pd.DataFrame(reliability_rows).sort_values(
        ["scope", "geometry", "method", "bin"], kind="mergesort"
    ).reset_index(drop=True)
    return blocks, units, summary, reliability, maps


def save_geometry_reconstruction_outputs(
    blocks: pd.DataFrame,
    units: pd.DataFrame,
    summary: pd.DataFrame,
    reliability: pd.DataFrame,
    maps: dict[str, np.ndarray],
) -> None:
    blocks.to_csv(DATA / "sparse_reconstruction_geometry_blocks.csv", index=False)
    units.to_csv(DATA / "sparse_reconstruction_geometry_units.csv", index=False)
    summary.to_csv(DATA / "sparse_reconstruction_geometry_summary.csv", index=False)
    reliability.to_csv(
        DATA / "sparse_reconstruction_geometry_calibration.csv", index=False
    )
    np.savez_compressed(DATA / "sparse_reconstruction_geometry_maps.npz", **maps)


def refresh_geometry_reconstruction() -> None:
    outputs = run_geometry_replicated_reconstruction(quick=False)
    save_geometry_reconstruction_outputs(*outputs)
    make_geometry_reconstruction_figure(outputs[1], outputs[3], outputs[4])
    print(
        "geometry-replicated reconstruction written\n",
        outputs[1].groupby("method")[[
            "masked_score_rmse", "masked_calibration_ece", "capacity_abs_error"
        ]].mean(),
    )
    print(
        "run_metadata.json must be refreshed after all targeted outputs with "
        "`python code/run_applied.py --refresh-run-metadata`"
    )


def _periodic_resample(field: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(
        resample(resample(field, n, axis=0), n, axis=1).real,
        dtype=float,
    )


def run_reconstruction_refinement(
    selected_horizon: float,
    quick: bool = False,
) -> pd.DataFrame:
    """Independent spatial and pseudo-time-step checks at fixed physical T."""
    rows: list[dict[str, float | int | str]] = []

    spatial_grids = (61, 81) if quick else (61, 81, 101)
    spatial_fields: dict[int, np.ndarray] = {}
    spatial_context: dict[int, tuple[SpectralGrid, np.ndarray, object, dict[str, float]]] = {}
    for n in spatial_grids:
        grid = reconstruction_grid(n)
        contacts = _build_contacts(grid, REPRESENTATIVE_SEED)
        confidence, data_forcing = compact_observation_fields(grid, contacts)
        initial, _ = screened_initial_state(grid, confidence, data_forcing)
        step = int(round(selected_horizon / PHASE_DT))
        state, history = run_reconstruction_history(
            grid,
            initial,
            confidence,
            data_forcing,
            "graph",
            step,
            dt=PHASE_DT,
        )
        field = terminal_average(history, step, window=TERMINAL_WINDOW)
        truth = lesion_score_field(grid.X, grid.Y)
        capacity, _ = capacity_metrics(score_to_fv_diffusivity(grid, field))
        spatial_fields[n] = field
        spatial_context[n] = (grid, truth, state, capacity)

    finest_n = max(spatial_grids)
    finest_field = spatial_fields[finest_n]
    for n in spatial_grids:
        grid, truth, state, capacity = spatial_context[n]
        reference = _periodic_resample(finest_field, n)
        difference = spatial_fields[n] - reference
        rows.append(
            {
                "study": "spatial",
                "n": n,
                "dx_mm": LENGTH / n,
                "pseudo_dt": PHASE_DT,
                "selected_horizon": selected_horizon,
                "terminal_window": TERMINAL_WINDOW * PHASE_DT,
                "field_rmse": float(np.sqrt(np.mean((spatial_fields[n] - truth) ** 2))),
                "h2k_error": grid.h2k_norm(spatial_fields[n] - truth),
                "l2_difference_from_finest": float(np.sqrt(np.mean(difference**2))),
                "h2k_difference_from_finest": grid.h2k_norm(difference),
                "gap_width_mm": ring_gap_width(
                    spatial_fields[n],
                    grid.X,
                    grid.Y,
                    CENTER,
                    RADIUS,
                    gap_angle=TRUTH_GAP_ANGLE,
                ),
                "normalized_capacity": capacity["normalized_capacity"],
                "final_laplacian_tail_energy": state.diagnostics[-1]["tail_laplacian"],
                "max_laplacian_tail_energy": max(
                    item["tail_laplacian"] for item in state.diagnostics
                ),
                "max_graph_state_residual": max(
                    item["state_residual"] for item in state.diagnostics
                ),
            }
        )

    grid = reconstruction_grid()
    contacts = _build_contacts(grid, REPRESENTATIVE_SEED)
    confidence, data_forcing = compact_observation_fields(grid, contacts)
    initial, _ = screened_initial_state(grid, confidence, data_forcing)
    pseudo_steps = (0.02, 0.01) if quick else (0.02, 0.01, 0.005)
    dt_fields: dict[float, np.ndarray] = {}
    dt_context: dict[float, tuple[np.ndarray, object, dict[str, float]]] = {}
    truth = lesion_score_field(grid.X, grid.Y)
    terminal_window_time = TERMINAL_WINDOW * PHASE_DT
    for pseudo_dt in pseudo_steps:
        step = int(round(selected_horizon / pseudo_dt))
        window = int(round(terminal_window_time / pseudo_dt))
        state, history = run_reconstruction_history(
            grid,
            initial,
            confidence,
            data_forcing,
            "graph",
            step,
            dt=pseudo_dt,
        )
        field = terminal_average(history, step, window=window)
        capacity, _ = capacity_metrics(score_to_fv_diffusivity(grid, field))
        dt_fields[pseudo_dt] = field
        dt_context[pseudo_dt] = (truth, state, capacity)

    finest_dt = min(pseudo_steps)
    for pseudo_dt in pseudo_steps:
        truth, state, capacity = dt_context[pseudo_dt]
        difference = dt_fields[pseudo_dt] - dt_fields[finest_dt]
        rows.append(
            {
                "study": "pseudo_time_step",
                "n": grid.nx,
                "dx_mm": grid.dx,
                "pseudo_dt": pseudo_dt,
                "selected_horizon": selected_horizon,
                "terminal_window": terminal_window_time,
                "field_rmse": float(np.sqrt(np.mean((dt_fields[pseudo_dt] - truth) ** 2))),
                "h2k_error": grid.h2k_norm(dt_fields[pseudo_dt] - truth),
                "l2_difference_from_finest": float(np.sqrt(np.mean(difference**2))),
                "h2k_difference_from_finest": grid.h2k_norm(difference),
                "gap_width_mm": ring_gap_width(
                    dt_fields[pseudo_dt],
                    grid.X,
                    grid.Y,
                    CENTER,
                    RADIUS,
                    gap_angle=TRUTH_GAP_ANGLE,
                ),
                "normalized_capacity": capacity["normalized_capacity"],
                "final_laplacian_tail_energy": state.diagnostics[-1]["tail_laplacian"],
                "max_laplacian_tail_energy": max(
                    item["tail_laplacian"] for item in state.diagnostics
                ),
                "max_graph_state_residual": max(
                    item["state_residual"] for item in state.diagnostics
                ),
            }
        )
    return pd.DataFrame(rows)


def _disk_stimulus(
    shape: tuple[int, int],
    centre: tuple[float, float],
    radius: float = STIMULUS_RADIUS,
    amplitude: float = STIMULUS_AMPLITUDE,
    duration: float = STIMULUS_DURATION,
) -> Callable[[float, np.ndarray, np.ndarray], np.ndarray]:
    def stimulus(t: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mask = (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= radius**2
        return amplitude * np.broadcast_to(mask, shape) * (t < duration)

    return stimulus


def ep_readout(
    diffusivity: np.ndarray,
    gap_angle: float,
    direction: str,
    t_end: float = 210.0,
    dt: float = PRODUCTION_EP_DT,
    face_average: str = "harmonic",
) -> tuple[dict[str, float | int | str], object]:
    n = diffusivity.shape[0]
    spacing = LENGTH / n
    coordinates = (np.arange(n) + 0.5) * spacing
    X, Y = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt((X - CENTER[0]) ** 2 + (Y - CENTER[1]) ** 2)
    angle = np.arctan2(Y - CENTER[1], X - CENTER[0])
    angular_distance = np.abs(_wrapped_angle(angle - gap_angle))
    direction_vector = np.array([np.cos(gap_angle), np.sin(gap_angle)])
    if direction == "exit":
        stimulus_centre = CENTER
        target = (
            (radius >= 20.0)
            & (radius <= 23.0)
            & (angular_distance <= TARGET_SECTOR_HALF_ANGLE)
        )
        probe_position = np.asarray(CENTER) + 21.5 * direction_vector
    elif direction == "entrance":
        stimulus_centre = tuple(np.asarray(CENTER) + 23.0 * direction_vector)
        target = (
            (radius >= 7.0)
            & (radius <= 10.0)
            & (angular_distance <= TARGET_SECTOR_HALF_ANGLE)
        )
        probe_position = np.asarray(CENTER) + 8.5 * direction_vector
    else:
        raise ValueError("direction must be 'exit' or 'entrance'")
    probe = (
        int(np.argmin(np.abs(coordinates - probe_position[0]))),
        int(np.argmin(np.abs(coordinates - probe_position[1]))),
    )
    parameters = ep_parameters(n, dt=dt, face_average=face_average)
    solution = solve_monodomain(
        np.zeros_like(diffusivity),
        parameters,
        t_end,
        _disk_stimulus(diffusivity.shape, stimulus_centre),
        trace_points={"target": probe},
        boundary="noflux",
        diffusivity_scale=diffusivity,
    )
    target_activation = solution.activation[target]
    activated = np.isfinite(target_activation)
    sector_activated_fraction = float(np.mean(activated))
    captured = sector_activated_fraction >= TARGET_CAPTURE_FRACTION
    raw_first_arrival = (
        float(np.nanmin(target_activation)) if np.any(activated) else np.nan
    )
    raw_probe_activation = float(solution.activation[probe])
    # Arrival-time summaries are conditional on the declared 80% sector-capture
    # event.  Peak voltage and activated fraction retain information about a
    # possible subthreshold or spatially partial response.
    first_arrival = raw_first_arrival if captured else np.nan
    probe_activation = raw_probe_activation if captured else np.nan
    output: dict[str, float | int | str] = {
        "crossing_time_ms": first_arrival,
        "sector_first_arrival_ms": first_arrival,
        "probe_activation_time_ms": probe_activation,
        "crossed_by_horizon": int(captured),
        "captured_by_horizon": int(captured),
        "target_activated_fraction": sector_activated_fraction,
        "sector_activated_fraction": sector_activated_fraction,
        "capture_fraction_threshold": TARGET_CAPTURE_FRACTION,
        "target_cell_count": int(np.sum(target)),
        "target_peak_voltage": float(np.max(solution.peak[target])),
        "probe_peak_voltage": float(np.max(solution.traces["target"])),
        "t_end_ms": t_end,
        "stimulus_radius_mm": STIMULUS_RADIUS,
        "stimulus_amplitude": STIMULUS_AMPLITUDE,
        "stimulus_duration_ms": STIMULUS_DURATION,
        "target_sector_half_angle_deg": np.rad2deg(TARGET_SECTOR_HALF_ANGLE),
        "face_average": face_average,
        **solution.diagnostics,
    }
    return output, solution


def _phase_lookup(
    phase: pd.DataFrame,
    width: float,
    gap_diffusivity: float,
    angle_deg: float,
) -> pd.Series:
    return phase.loc[
        np.isclose(phase["gap_width_mm"], width)
        & np.isclose(phase["gap_diffusivity_fraction"], gap_diffusivity)
        & np.isclose(phase["gap_angle_deg"], angle_deg)
    ].iloc[0]


def _matched_capacity_cases(phase: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """Apply a declared nearest-capacity rule without using EP outcomes."""
    candidates: list[tuple[float, tuple[float, ...], int, int]] = []
    records = list(phase.reset_index(drop=True).iterrows())
    for i, first in records:
        for j, second in records[i + 1 :]:
            width_separation = abs(
                float(first["gap_width_mm"]) - float(second["gap_width_mm"])
            )
            angle_separation = abs(
                float(first["gap_angle_deg"]) - float(second["gap_angle_deg"])
            )
            eta_ratio = max(
                float(first["gap_diffusivity_fraction"]),
                float(second["gap_diffusivity_fraction"]),
            ) / min(
                float(first["gap_diffusivity_fraction"]),
                float(second["gap_diffusivity_fraction"]),
            )
            if width_separation < 3.0 or angle_separation < 45.0 or eta_ratio < 3.0:
                continue
            c_first = float(first["normalized_capacity"])
            c_second = float(second["normalized_capacity"])
            relative_difference = abs(c_first - c_second) / max(
                0.5 * (c_first + c_second), 1.0e-14
            )
            tie_break = (
                float(first["gap_width_mm"]),
                float(first["gap_diffusivity_fraction"]),
                float(first["gap_angle_deg"]),
                float(second["gap_width_mm"]),
                float(second["gap_diffusivity_fraction"]),
                float(second["gap_angle_deg"]),
            )
            candidates.append((relative_difference, tie_break, int(i), int(j)))
    if not candidates:
        raise RuntimeError("the capacity grid has no admissible matched pair")
    relative_difference, _, i, j = min(candidates, key=lambda item: (item[0], item[1]))
    return phase.iloc[i], phase.iloc[j], relative_difference


def run_matched_capacity_refinement(
    phase: pd.DataFrame,
    quick: bool = False,
) -> pd.DataFrame:
    """Refine both outcome-blind capacity-matched cases in space and time.

    The pair is reselected from the supplied passive-capacity table without using
    EP results.  Spatial runs pace both directions on each grid at the production
    time step; temporal runs pace both directions on the production grid.  The
    saved pair difference is recomputed independently at every spatial grid.
    """
    first, second, _ = _matched_capacity_cases(phase)
    matched_cases = (("A", first), ("B", second))
    spatial_grids = (81,) if quick else (81, 121, 161)
    temporal_steps = (PRODUCTION_EP_DT,) if quick else (0.01, 0.02, 0.04)
    horizon = 90.0 if quick else 210.0
    rows: list[dict[str, float | int | str]] = []

    def append_run(
        *,
        quantity: str,
        label: str,
        matched: pd.Series,
        n_grid: int,
        dt: float,
        direction: str,
    ) -> None:
        width = float(matched["gap_width_mm"])
        gap_diffusivity = float(matched["gap_diffusivity_fraction"])
        angle_deg = float(matched["gap_angle_deg"])
        angle = np.deg2rad(angle_deg)
        scale = analytic_subcell_diffusivity(
            n_grid, width, gap_diffusivity, angle, subcells=SUBCELLS_PER_AXIS
        )
        capacity, _ = capacity_metrics(scale)
        result, _ = ep_readout(
            scale,
            angle,
            direction,
            t_end=horizon,
            dt=dt,
        )
        rows.append(
            {
                "quantity": quantity,
                "regime": f"matched_{label}",
                "matched_case": label,
                "matched_pair_relative_capacity_difference": np.nan,
                "pacing_direction": direction,
                "gap_width_mm": width,
                "gap_diffusivity_fraction": gap_diffusivity,
                "gap_angle_deg": angle_deg,
                "n": n_grid,
                "dx_mm": LENGTH / n_grid,
                "dt_ms": dt,
                "subcells_per_axis": SUBCELLS_PER_AXIS,
                "capacity": capacity["capacity"],
                "normalized_capacity": capacity["normalized_capacity"],
                "capacity_inner_flux": capacity["capacity_inner_flux"],
                "capacity_outer_flux": capacity["capacity_outer_flux"],
                "capacity_energy": capacity["capacity_energy"],
                "capacity_relative_residual": capacity["capacity_relative_residual"],
                "capacity_flux_energy_defect": capacity["capacity_flux_energy_defect"],
                "crossing_time_ms": result["crossing_time_ms"],
                "crossed_by_horizon": result["crossed_by_horizon"],
                "captured_by_horizon": result["captured_by_horizon"],
                "probe_activation_time_ms": result["probe_activation_time_ms"],
                "sector_activated_fraction": result["sector_activated_fraction"],
                "t_end_ms": result["t_end_ms"],
                "face_average": result["face_average"],
                "target_peak_voltage": result["target_peak_voltage"],
                "cfl": result["cfl"],
            }
        )

    for n_grid in spatial_grids:
        for label, matched in matched_cases:
            for direction in ("exit", "entrance"):
                append_run(
                    quantity="matched_EP_space",
                    label=label,
                    matched=matched,
                    n_grid=n_grid,
                    dt=PRODUCTION_EP_DT,
                    direction=direction,
                )

    for dt in temporal_steps:
        for label, matched in matched_cases:
            for direction in ("exit", "entrance"):
                append_run(
                    quantity="matched_EP_time",
                    label=label,
                    matched=matched,
                    n_grid=PRODUCTION_EP_N,
                    dt=dt,
                    direction=direction,
                )

    frame = pd.DataFrame(rows)
    group_columns = {
        "matched_EP_space": "n",
        "matched_EP_time": "dt_ms",
    }
    for quantity, refinement_column in group_columns.items():
        subset = frame.loc[frame["quantity"] == quantity]
        for value, group in subset.groupby(refinement_column):
            capacities = group.groupby("matched_case")["normalized_capacity"].first()
            if set(capacities.index) != {"A", "B"}:
                raise RuntimeError("matched refinement lost one member of the fixed pair")
            difference = abs(float(capacities["A"] - capacities["B"])) / float(
                0.5 * (capacities["A"] + capacities["B"])
            )
            mask = (frame["quantity"] == quantity) & np.isclose(
                frame[refinement_column], value
            )
            frame.loc[mask, "matched_pair_relative_capacity_difference"] = difference
    return frame


def run_capacity_phase_diagram(
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    n = 81 if quick else 121
    widths = (3.0, 6.0) if quick else (3.0, 4.5, 6.0, 9.0)
    diffusivities = (
        (0.30, 0.03, 0.003)
        if quick
        else (1.0, 0.30, 0.10, 0.03, 0.01, 0.003)
    )
    angles_deg = (0.0, 45.0, 90.0)

    rows: list[dict[str, float]] = []
    representatives: dict[str, np.ndarray] = {}
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        for width in widths:
            for gap_diffusivity in diffusivities:
                scale = analytic_subcell_diffusivity(n, width, gap_diffusivity, angle)
                capacity, potential = capacity_metrics(scale)
                rows.append(
                    {
                        "n": n,
                        "dx_mm": LENGTH / n,
                        "gap_width_mm": width,
                        "gap_diffusivity_fraction": gap_diffusivity,
                        "nominal_gap_diffusivity_fraction": gap_diffusivity,
                        "realised_centerline_gap_diffusivity_fraction": (
                            realised_centerline_gap_diffusivity(
                                width, gap_diffusivity, angle
                            )
                        ),
                        "gap_angle_deg": angle_deg,
                        "gap_orientation_relative_to_fiber_deg": angle_deg,
                        "fiber_axis": "x",
                        "face_average": "harmonic",
                        "lesion_width_mm": LESION_WIDTH,
                        "transition_scale_mm": TRANSITION,
                        "d_long_mm2_per_ms": ep_parameters(n).d_long,
                        "d_trans_mm2_per_ms": ep_parameters(n).d_trans,
                        "cells_across_zero_level_gap": width / (LENGTH / n),
                        "cells_across_transition": TRANSITION / (LENGTH / n),
                        **capacity,
                    }
                )
                if (
                    np.isclose(angle_deg, 45.0)
                    and np.isclose(width, 6.0)
                    and np.isclose(gap_diffusivity, 0.03)
                ):
                    representatives["diffusivity"] = scale
                    representatives["potential"] = potential
    phase = pd.DataFrame(rows)
    phase["subcells_per_axis"] = SUBCELLS_PER_AXIS

    # The EP cases are a predeclared diagonal stratification of the inexpensive
    # conductance grid, not a Cartesian search tuned to breakthrough labels.
    stratified_cases = (
        ((3.0, 0.30), (6.0, 0.03))
        if quick
        else ((3.0, 0.30), (6.0, 0.03), (9.0, 0.003))
    )
    ep_rows: list[dict[str, float | int | str]] = []
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        for width, gap_diffusivity in stratified_cases:
            scale = analytic_subcell_diffusivity(n, width, gap_diffusivity, angle)
            match = _phase_lookup(phase, width, gap_diffusivity, angle_deg)
            for direction in ("exit", "entrance"):
                result, solution = ep_readout(
                    scale,
                    angle,
                    direction,
                    t_end=90.0 if quick else 210.0,
                    dt=PRODUCTION_EP_DT,
                )
                ep_rows.append(
                    {
                        "experiment": "stratified",
                        "case_id": f"w{width:g}_eta{gap_diffusivity:g}_a{angle_deg:g}",
                        "matched_case": "",
                        "matched_pair_relative_capacity_difference": np.nan,
                        "face_average": "harmonic",
                        "n": n,
                        "dx_mm": LENGTH / n,
                        "dt_ms": PRODUCTION_EP_DT,
                        "gap_width_mm": width,
                        "gap_diffusivity_fraction": gap_diffusivity,
                        "gap_angle_deg": angle_deg,
                        "pacing_direction": direction,
                        "capacity": float(match["capacity"]),
                        "normalized_capacity": float(match["normalized_capacity"]),
                        "capacity_inner_flux": float(match["capacity_inner_flux"]),
                        "capacity_outer_flux": float(match["capacity_outer_flux"]),
                        "capacity_energy": float(match["capacity_energy"]),
                        "capacity_relative_residual": float(
                            match["capacity_relative_residual"]
                        ),
                        "capacity_flux_energy_defect": float(
                            match["capacity_flux_energy_defect"]
                        ),
                        **result,
                    }
                )
                if (
                    np.isclose(angle_deg, 45.0)
                    and np.isclose(width, 6.0)
                    and np.isclose(gap_diffusivity, 0.03)
                ):
                    representatives[f"{direction}_activation"] = solution.activation
                    representatives[f"{direction}_peak"] = solution.peak

    # A complete-ring negative control tests numerical leakage in both pacing
    # directions.  Its angle merely places the exterior stimulus.
    control_angle_deg = 45.0
    control_angle = np.deg2rad(control_angle_deg)
    complete_ring = analytic_subcell_diffusivity(
        n, 0.0, ETA_MIN, control_angle
    )
    control_capacity, _ = capacity_metrics(complete_ring)
    for direction in ("exit", "entrance"):
        result, _ = ep_readout(
            complete_ring,
            control_angle,
            direction,
            t_end=90.0 if quick else 210.0,
            dt=PRODUCTION_EP_DT,
        )
        ep_rows.append(
            {
                "experiment": "complete_ring_control",
                "case_id": "complete_ring",
                "matched_case": "",
                "matched_pair_relative_capacity_difference": np.nan,
                "face_average": "harmonic",
                "n": n,
                "dx_mm": LENGTH / n,
                "dt_ms": PRODUCTION_EP_DT,
                "gap_width_mm": 0.0,
                "gap_diffusivity_fraction": ETA_MIN,
                "gap_angle_deg": control_angle_deg,
                "pacing_direction": direction,
                "capacity": control_capacity["capacity"],
                "normalized_capacity": control_capacity["normalized_capacity"],
                "capacity_inner_flux": control_capacity["capacity_inner_flux"],
                "capacity_outer_flux": control_capacity["capacity_outer_flux"],
                "capacity_energy": control_capacity["capacity_energy"],
                "capacity_relative_residual": control_capacity[
                    "capacity_relative_residual"
                ],
                "capacity_flux_energy_defect": control_capacity[
                    "capacity_flux_energy_defect"
                ],
                **result,
            }
        )

    # Select geometrically and materially different cases by a declared
    # nearest-capacity rule, then compare EP without consulting EP outcomes.
    first, second, pair_difference = _matched_capacity_cases(phase)
    for label, matched in (("A", first), ("B", second)):
        width = float(matched["gap_width_mm"])
        gap_diffusivity = float(matched["gap_diffusivity_fraction"])
        angle_deg = float(matched["gap_angle_deg"])
        angle = np.deg2rad(angle_deg)
        scale = analytic_subcell_diffusivity(n, width, gap_diffusivity, angle)
        for direction in ("exit", "entrance"):
            result, _ = ep_readout(
                scale,
                angle,
                direction,
                t_end=90.0 if quick else 210.0,
                dt=PRODUCTION_EP_DT,
            )
            ep_rows.append(
                {
                    "experiment": "matched_capacity",
                    "case_id": f"matched_{label}",
                    "matched_case": label,
                    "matched_pair_relative_capacity_difference": pair_difference,
                    "face_average": "harmonic",
                    "n": n,
                    "dx_mm": LENGTH / n,
                    "dt_ms": PRODUCTION_EP_DT,
                    "gap_width_mm": width,
                    "gap_diffusivity_fraction": gap_diffusivity,
                    "gap_angle_deg": angle_deg,
                    "pacing_direction": direction,
                    "capacity": float(matched["capacity"]),
                    "normalized_capacity": float(matched["normalized_capacity"]),
                    "capacity_inner_flux": float(matched["capacity_inner_flux"]),
                    "capacity_outer_flux": float(matched["capacity_outer_flux"]),
                    "capacity_energy": float(matched["capacity_energy"]),
                    "capacity_relative_residual": float(
                        matched["capacity_relative_residual"]
                    ),
                    "capacity_flux_energy_defect": float(
                        matched["capacity_flux_energy_defect"]
                    ),
                    **result,
                }
            )

    # High-contrast face-mean sensitivity addresses the main discretisation
    # failure mode directly while changing no biological parameter.
    sensitivity_width = 6.0
    sensitivity_eta = 0.003
    sensitivity_angle_deg = 45.0
    sensitivity_angle = np.deg2rad(sensitivity_angle_deg)
    sensitivity_scale = analytic_subcell_diffusivity(
        n, sensitivity_width, sensitivity_eta, sensitivity_angle
    )
    for face_average in ("harmonic", "arithmetic"):
        sensitivity_capacity, _ = capacity_metrics(
            sensitivity_scale, face_average=face_average
        )
        result, _ = ep_readout(
            sensitivity_scale,
            sensitivity_angle,
            "exit",
            t_end=90.0 if quick else 210.0,
            dt=PRODUCTION_EP_DT,
            face_average=face_average,
        )
        ep_rows.append(
            {
                "experiment": "face_mean_sensitivity",
                "case_id": f"face_mean_{face_average}",
                "matched_case": "",
                "matched_pair_relative_capacity_difference": np.nan,
                "face_average": face_average,
                "n": n,
                "dx_mm": LENGTH / n,
                "dt_ms": PRODUCTION_EP_DT,
                "gap_width_mm": sensitivity_width,
                "gap_diffusivity_fraction": sensitivity_eta,
                "gap_angle_deg": sensitivity_angle_deg,
                "pacing_direction": "exit",
                "capacity": sensitivity_capacity["capacity"],
                "normalized_capacity": sensitivity_capacity["normalized_capacity"],
                "capacity_inner_flux": sensitivity_capacity["capacity_inner_flux"],
                "capacity_outer_flux": sensitivity_capacity["capacity_outer_flux"],
                "capacity_energy": sensitivity_capacity["capacity_energy"],
                "capacity_relative_residual": sensitivity_capacity[
                    "capacity_relative_residual"
                ],
                "capacity_flux_energy_defect": sensitivity_capacity[
                    "capacity_flux_energy_defect"
                ],
                **result,
            }
        )
    ep_frame = pd.DataFrame(ep_rows)
    ep_frame["nominal_gap_diffusivity_fraction"] = ep_frame[
        "gap_diffusivity_fraction"
    ]
    ep_frame["realised_centerline_gap_diffusivity_fraction"] = [
        realised_centerline_gap_diffusivity(
            float(width), float(diffusivity), np.deg2rad(float(angle_deg))
        )
        for width, diffusivity, angle_deg in zip(
            ep_frame["gap_width_mm"],
            ep_frame["gap_diffusivity_fraction"],
            ep_frame["gap_angle_deg"],
        )
    ]
    ep_frame["subcells_per_axis"] = SUBCELLS_PER_AXIS
    ep_frame["fiber_axis"] = "x"
    ep_frame["lesion_width_mm"] = LESION_WIDTH
    ep_frame["transition_scale_mm"] = TRANSITION
    ep_frame["d_long_mm2_per_ms"] = ep_parameters(n).d_long
    ep_frame["d_trans_mm2_per_ms"] = ep_parameters(n).d_trans

    # Capacity refinement on the representative case and a small EP refinement
    # pair well inside the conducting and low-conductance regimes.
    refinement_rows: list[dict[str, float | int | str]] = []
    capacity_grids = (61, 81) if quick else (61, 91, 121, 161)
    for n_refine in capacity_grids:
        scale = analytic_subcell_diffusivity(
            n_refine, 6.0, 0.03, np.deg2rad(45.0), subcells=3
        )
        capacity, _ = capacity_metrics(scale)
        refinement_rows.append(
            {
                "quantity": "capacity",
                "regime": "representative",
                "n": n_refine,
                "dx_mm": LENGTH / n_refine,
                "dt_ms": np.nan,
                "subcells_per_axis": SUBCELLS_PER_AXIS,
                "capacity": capacity["capacity"],
                "normalized_capacity": capacity["normalized_capacity"],
                "capacity_inner_flux": capacity["capacity_inner_flux"],
                "capacity_outer_flux": capacity["capacity_outer_flux"],
                "capacity_energy": capacity["capacity_energy"],
                "crossing_time_ms": np.nan,
                "crossed_by_horizon": np.nan,
                "capacity_relative_residual": capacity["capacity_relative_residual"],
                "capacity_flux_energy_defect": capacity["capacity_flux_energy_defect"],
            }
        )
    ep_grids = (81,) if quick else (81, 121, 161)
    for regime, gap_diffusivity in (("conducting", 0.30), ("low_conductance", 0.003)):
        for n_refine in ep_grids:
            scale = analytic_subcell_diffusivity(
                n_refine, 6.0, gap_diffusivity, np.deg2rad(45.0), subcells=3
            )
            result, _ = ep_readout(
                scale,
                np.deg2rad(45.0),
                "exit",
                t_end=90.0 if quick else 210.0,
                dt=PRODUCTION_EP_DT,
            )
            capacity, _ = capacity_metrics(scale)
            refinement_rows.append(
                {
                    "quantity": "EP",
                    "regime": regime,
                    "n": n_refine,
                    "dx_mm": LENGTH / n_refine,
                    "dt_ms": PRODUCTION_EP_DT,
                    "subcells_per_axis": SUBCELLS_PER_AXIS,
                    "capacity": capacity["capacity"],
                    "normalized_capacity": capacity["normalized_capacity"],
                    "capacity_inner_flux": capacity["capacity_inner_flux"],
                    "capacity_outer_flux": capacity["capacity_outer_flux"],
                    "capacity_energy": capacity["capacity_energy"],
                    "crossing_time_ms": result["crossing_time_ms"],
                    "crossed_by_horizon": result["crossed_by_horizon"],
                    "captured_by_horizon": result["captured_by_horizon"],
                    "probe_activation_time_ms": result["probe_activation_time_ms"],
                    "sector_activated_fraction": result["sector_activated_fraction"],
                    "t_end_ms": result["t_end_ms"],
                    "face_average": result["face_average"],
                    "capacity_relative_residual": capacity["capacity_relative_residual"],
                    "capacity_flux_energy_defect": capacity["capacity_flux_energy_defect"],
                    "target_peak_voltage": result["target_peak_voltage"],
                    "cfl": result["cfl"],
                }
            )

    # Independent temporal refinement at fixed spatial grid and coefficient.
    dt_values = (PRODUCTION_EP_DT,) if quick else (0.01, 0.02, 0.04)
    fixed_scale = analytic_subcell_diffusivity(
        PRODUCTION_EP_N, 6.0, 0.30, np.deg2rad(45.0), subcells=3
    )
    fixed_capacity, _ = capacity_metrics(fixed_scale)
    for dt in dt_values:
        result, _ = ep_readout(
            fixed_scale,
            np.deg2rad(45.0),
            "exit",
            t_end=90.0 if quick else 210.0,
            dt=dt,
        )
        refinement_rows.append(
            {
                "quantity": "EP_dt",
                "regime": "conducting_fixed_n",
                "n": PRODUCTION_EP_N,
                "dx_mm": LENGTH / PRODUCTION_EP_N,
                "dt_ms": dt,
                "subcells_per_axis": SUBCELLS_PER_AXIS,
                "capacity": fixed_capacity["capacity"],
                "normalized_capacity": fixed_capacity["normalized_capacity"],
                "capacity_inner_flux": fixed_capacity["capacity_inner_flux"],
                "capacity_outer_flux": fixed_capacity["capacity_outer_flux"],
                "capacity_energy": fixed_capacity["capacity_energy"],
                "crossing_time_ms": result["crossing_time_ms"],
                "crossed_by_horizon": result["crossed_by_horizon"],
                "captured_by_horizon": result["captured_by_horizon"],
                "probe_activation_time_ms": result["probe_activation_time_ms"],
                "sector_activated_fraction": result["sector_activated_fraction"],
                "t_end_ms": result["t_end_ms"],
                "face_average": result["face_average"],
                "capacity_relative_residual": fixed_capacity[
                    "capacity_relative_residual"
                ],
                "capacity_flux_energy_defect": fixed_capacity[
                    "capacity_flux_energy_defect"
                ],
                "target_peak_voltage": result["target_peak_voltage"],
                "cfl": result["cfl"],
            }
        )
    refinement_rows.extend(
        run_matched_capacity_refinement(phase, quick=quick).to_dict("records")
    )
    return phase, ep_frame, pd.DataFrame(refinement_rows), representatives


def refresh_matched_capacity_refinement() -> None:
    """Recompute only the matched-pair refinement from committed capacity data."""
    phase_path = DATA / "pvi_capacity_phase_diagram.csv"
    refinement_path = DATA / "pvi_capacity_refinement.csv"
    if not phase_path.exists() or not refinement_path.exists():
        raise FileNotFoundError(
            "run the publication applied suite before the targeted matched refinement"
        )
    phase = pd.read_csv(phase_path)
    refinement = pd.read_csv(refinement_path)
    refinement = refinement.loc[
        ~refinement["quantity"].isin({"matched_EP_space", "matched_EP_time"})
    ].copy()
    matched = run_matched_capacity_refinement(phase, quick=False)
    pd.concat([refinement, matched], ignore_index=True, sort=False).to_csv(
        refinement_path, index=False
    )
    print(
        "matched-pair refinement written\n",
        matched.groupby(["quantity", "matched_case", "pacing_direction"])[
            ["crossed_by_horizon", "target_peak_voltage"]
        ].agg(["min", "max"]),
    )
    print(
        "run_metadata.json must be refreshed after all targeted outputs with "
        "`python code/run_applied.py --refresh-run-metadata`"
    )


def run_end_to_end_ensemble(
    fields_by_seed: dict[int, dict[str, np.ndarray]],
    sparse_results: pd.DataFrame,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    grid = reconstruction_grid()
    truth = lesion_score_field(grid.X, grid.Y)
    truth_diffusivity = score_to_fv_diffusivity(grid, truth)
    truth_capacity, truth_potential = capacity_metrics(truth_diffusivity)
    seeds = sorted(fields_by_seed)
    ep_seeds = (
        {REPRESENTATIVE_SEED}
        if quick
        else set(seeds).intersection(EP_ENSEMBLE_SEEDS)
    )

    truth_ep: dict[str, dict[str, float | int]] = {}
    truth_solutions: dict[str, object] = {}
    for direction in ("exit", "entrance"):
        result, solution = ep_readout(
            truth_diffusivity,
            TRUTH_GAP_ANGLE,
            direction,
            t_end=90.0 if quick else 210.0,
            dt=PRODUCTION_EP_DT,
        )
        truth_ep[direction] = result
        truth_solutions[direction] = solution

    rows: list[dict[str, float | int | str]] = []
    saved: dict[str, np.ndarray] = {
        "truth_diffusivity": truth_diffusivity,
        "truth_potential": truth_potential,
        "truth_exit_activation": truth_solutions["exit"].activation,
        "truth_exit_peak": truth_solutions["exit"].peak,
        "truth_entrance_activation": truth_solutions["entrance"].activation,
        "truth_entrance_peak": truth_solutions["entrance"].peak,
    }
    for seed in seeds:
        fields = fields_by_seed[seed]
        for method, field in fields.items():
            diffusivity = score_to_fv_diffusivity(grid, field)
            capacity, potential = capacity_metrics(diffusivity)
            base = {
                "seed": seed,
                "method": method,
                "ep_included": int(seed in ep_seeds and method in ("passive", "graph")),
                "ep_n": PRODUCTION_EP_N,
                "ep_dx_mm": LENGTH / PRODUCTION_EP_N,
                "ep_dt_ms": PRODUCTION_EP_DT,
                "subcells_per_axis": SUBCELLS_PER_AXIS,
                "face_average": "harmonic",
                "capacity": capacity["capacity"],
                "normalized_capacity": capacity["normalized_capacity"],
                "capacity_inner_flux": capacity["capacity_inner_flux"],
                "capacity_outer_flux": capacity["capacity_outer_flux"],
                "capacity_energy": capacity["capacity_energy"],
                "capacity_relative_residual": capacity["capacity_relative_residual"],
                "capacity_flux_energy_defect": capacity[
                    "capacity_flux_energy_defect"
                ],
                "truth_normalized_capacity": truth_capacity["normalized_capacity"],
                "capacity_error": capacity["normalized_capacity"]
                - truth_capacity["normalized_capacity"],
                "capacity_abs_error": abs(
                    capacity["normalized_capacity"] - truth_capacity["normalized_capacity"]
                ),
                "rmse": float(
                    sparse_results.loc[
                        (sparse_results["seed"] == seed)
                        & (sparse_results["method"] == method),
                        "rmse",
                    ].iloc[0]
                ),
                "exit_crossing_time_ms": np.nan,
                "exit_crossed_by_horizon": np.nan,
                "exit_peak_voltage": np.nan,
                "exit_probe_activation_time_ms": np.nan,
                "exit_sector_activated_fraction": np.nan,
                "entrance_crossing_time_ms": np.nan,
                "entrance_crossed_by_horizon": np.nan,
                "entrance_peak_voltage": np.nan,
                "entrance_probe_activation_time_ms": np.nan,
                "entrance_sector_activated_fraction": np.nan,
                "capture_fraction_threshold": TARGET_CAPTURE_FRACTION,
            }
            for direction in ("exit", "entrance"):
                base[f"truth_{direction}_crossing_time_ms"] = truth_ep[direction][
                    "crossing_time_ms"
                ]
                base[f"truth_{direction}_crossed_by_horizon"] = truth_ep[direction][
                    "crossed_by_horizon"
                ]
                base[f"truth_{direction}_peak_voltage"] = truth_ep[direction][
                    "target_peak_voltage"
                ]
                base[f"truth_{direction}_probe_activation_time_ms"] = truth_ep[
                    direction
                ]["probe_activation_time_ms"]
                base[f"truth_{direction}_sector_activated_fraction"] = truth_ep[
                    direction
                ]["sector_activated_fraction"]
                base[f"{direction}_crossing_class_correct"] = np.nan
                base[f"{direction}_crossing_time_error_ms"] = np.nan
                base[f"{direction}_peak_voltage_error"] = np.nan

            if seed in ep_seeds and method in ("passive", "graph"):
                for direction in ("exit", "entrance"):
                    result, solution = ep_readout(
                        diffusivity,
                        TRUTH_GAP_ANGLE,
                        direction,
                        t_end=90.0 if quick else 210.0,
                        dt=PRODUCTION_EP_DT,
                    )
                    base[f"{direction}_crossing_time_ms"] = result["crossing_time_ms"]
                    base[f"{direction}_crossed_by_horizon"] = result["crossed_by_horizon"]
                    base[f"{direction}_peak_voltage"] = result["target_peak_voltage"]
                    base[f"{direction}_probe_activation_time_ms"] = result[
                        "probe_activation_time_ms"
                    ]
                    base[f"{direction}_sector_activated_fraction"] = result[
                        "sector_activated_fraction"
                    ]
                    base[f"{direction}_crossing_class_correct"] = int(
                        result["crossed_by_horizon"]
                        == truth_ep[direction]["crossed_by_horizon"]
                    )
                    if (
                        result["crossed_by_horizon"]
                        and truth_ep[direction]["crossed_by_horizon"]
                    ):
                        base[f"{direction}_crossing_time_error_ms"] = (
                            result["crossing_time_ms"]
                            - truth_ep[direction]["crossing_time_ms"]
                        )
                    base[f"{direction}_peak_voltage_error"] = (
                        result["target_peak_voltage"]
                        - truth_ep[direction]["target_peak_voltage"]
                    )
                    if seed == REPRESENTATIVE_SEED and method == "graph":
                        saved[f"graph_{direction}_activation"] = solution.activation
                        saved[f"graph_{direction}_peak"] = solution.peak
            if seed == REPRESENTATIVE_SEED and method == "graph":
                saved["graph_score"] = field
                saved["graph_diffusivity"] = diffusivity
                saved["graph_potential"] = potential
            rows.append(base)
    frame = pd.DataFrame(rows)

    rng = np.random.default_rng(20260903)
    summary_rows: list[dict[str, float | int | str]] = []
    for method, group in frame.groupby("method"):
        for metric in ("rmse", "capacity_abs_error", "normalized_capacity"):
            values = group[metric].dropna().to_numpy(dtype=float)
            means = np.mean(rng.choice(values, size=(5000, len(values)), replace=True), axis=1)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "interval_type": "bootstrap_percentile_mean",
                    "lower_95": float(np.quantile(means, 0.025)),
                    "upper_95": float(np.quantile(means, 0.975)),
                }
            )

    # Paired graph-minus-passive effects are reported without assuming that the
    # active graph term must improve every metric.
    for metric in ("rmse", "capacity_abs_error", "normalized_capacity"):
        pivot = frame.pivot(index="seed", columns="method", values=metric).dropna()
        differences = (pivot["graph"] - pivot["passive"]).to_numpy(dtype=float)
        means = np.mean(
            rng.choice(differences, size=(5000, len(differences)), replace=True), axis=1
        )
        summary_rows.append(
            {
                "method": "graph_minus_passive",
                "metric": metric,
                "n": len(differences),
                "mean": float(np.mean(differences)),
                "sample_sd": float(np.std(differences, ddof=1)),
                "interval_type": "paired_bootstrap_percentile_mean",
                "lower_95": float(np.quantile(means, 0.025)),
                "upper_95": float(np.quantile(means, 0.975)),
            }
        )

    z = 1.959963984540054
    for method in ("passive", "graph"):
        subset = frame.loc[frame["method"] == method]
        for direction in ("exit", "entrance"):
            crossed = subset[f"{direction}_crossed_by_horizon"].dropna().to_numpy(dtype=float)
            n_crossed = len(crossed)
            if n_crossed:
                proportion = float(np.mean(crossed))
                denominator = 1.0 + z**2 / n_crossed
                centre = (proportion + z**2 / (2.0 * n_crossed)) / denominator
                half_width = z * math.sqrt(
                    proportion * (1.0 - proportion) / n_crossed
                    + z**2 / (4.0 * n_crossed**2)
                ) / denominator
                lower, upper = centre - half_width, centre + half_width
            else:
                proportion = lower = upper = np.nan
            summary_rows.append(
                {
                    "method": method,
                    "metric": f"{direction}_crossing_fraction",
                    "n": n_crossed,
                    "mean": proportion,
                    "sample_sd": float(np.std(crossed, ddof=1)) if n_crossed > 1 else 0.0,
                    "interval_type": "Wilson_score",
                    "lower_95": lower,
                    "upper_95": upper,
                }
            )

    # Transfer the same representative reconstructed score to several FV grids.
    # The pseudo-time reconstruction is unchanged; only the downstream EP mesh
    # varies, with a fixed time step.
    representative_score = fields_by_seed[REPRESENTATIVE_SEED]["graph"]
    grid_rows: list[dict[str, float | int]] = []
    ep_grids = (81,) if quick else (81, 121, 161)
    for n_refine in ep_grids:
        diffusivity = score_to_fv_diffusivity(grid, representative_score, n_refine)
        capacity, _ = capacity_metrics(diffusivity)
        result, _ = ep_readout(
            diffusivity,
            TRUTH_GAP_ANGLE,
            "exit",
            t_end=90.0 if quick else 210.0,
            dt=PRODUCTION_EP_DT,
        )
        grid_rows.append(
            {
                "seed": REPRESENTATIVE_SEED,
                "n": n_refine,
                "dx_mm": LENGTH / n_refine,
                "dt_ms": PRODUCTION_EP_DT,
                "normalized_capacity": capacity["normalized_capacity"],
                "crossing_time_ms": result["crossing_time_ms"],
                "crossed_by_horizon": result["crossed_by_horizon"],
                "target_peak_voltage": result["target_peak_voltage"],
                "cfl": result["cfl"],
                "capacity_relative_residual": capacity["capacity_relative_residual"],
                "capacity_flux_energy_defect": capacity["capacity_flux_energy_defect"],
            }
        )
    return frame, pd.DataFrame(summary_rows), pd.DataFrame(grid_rows), saved


def make_geometry_reconstruction_figure(
    units: pd.DataFrame,
    reliability: pd.DataFrame,
    maps: dict[str, np.ndarray],
) -> None:
    """Figure 4: prespecified geometries, spatial block and continuous scores."""

    figure = plt.figure(figsize=(7.15, 5.05), constrained_layout=True)
    outer = figure.add_gridspec(2, 1, height_ratios=(1.0, 1.35))
    upper = outer[0].subgridspec(1, len(GEOMETRY_SPECS), wspace=0.08)
    titles = {
        "complete_ring": "complete ring",
        "narrow_gap": "3-mm gap",
        "wide_gap": "9-mm gap",
        "two_gaps": "two gaps",
        "oblique_gap": "oblique gap",
    }
    image = None
    for index, specification in enumerate(GEOMETRY_SPECS):
        geometry = str(specification["geometry"])
        if f"truth_{geometry}" not in maps:
            continue
        axis = figure.add_subplot(upper[0, index])
        image = axis.imshow(
            maps[f"truth_{geometry}"].T,
            origin="lower",
            extent=(0, LENGTH, 0, LENGTH),
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
            aspect="equal",
        )
        axis.set_title(f"({chr(ord('a') + index)}) {titles[geometry]}", fontsize=7.2)
        axis.set_xticks(())
        axis.set_yticks(())
    if image is not None:
        figure.colorbar(
            image,
            ax=figure.axes[: len(GEOMETRY_SPECS)],
            shrink=0.62,
            pad=0.01,
            label="hidden continuous score",
        )

    lower = outer[1].subgridspec(1, 3, wspace=0.30)
    confidence_axis = figure.add_subplot(lower[0, 0])
    confidence = maps["representative_confidence"]
    plotted = np.ma.log10(np.ma.masked_less_equal(confidence, 0.0))
    confidence_axis.set_facecolor("white")
    confidence_image = confidence_axis.imshow(
        plotted.T,
        origin="lower",
        extent=(0, LENGTH, 0, LENGTH),
        cmap="viridis",
        aspect="equal",
    )
    confidence_axis.contour(
        maps["representative_holdout_mask"].T,
        levels=[0.5],
        colors="#c83e4d",
        linewidths=1.0,
        origin="lower",
        extent=(0, LENGTH, 0, LENGTH),
    )
    confidence_axis.scatter(
        maps["representative_train_x"],
        maps["representative_train_y"],
        s=1.0,
        c="black",
        alpha=0.35,
        linewidths=0,
    )
    confidence_axis.set_title(r"(f) fixed spatial blackout", fontsize=7.5)
    confidence_axis.set_xlabel("x (mm)")
    confidence_axis.set_ylabel("y (mm)")
    figure.colorbar(
        confidence_image,
        ax=confidence_axis,
        shrink=0.65,
        pad=0.02,
        label=r"$\log_{10}\lambda$",
    )

    error_axis = figure.add_subplot(lower[0, 1])
    method_order = ("screened", "passive", "graph")
    method_x = np.arange(len(method_order))
    colours = {
        "complete_ring": "#4c78a8",
        "narrow_gap": "#f58518",
        "wide_gap": "#54a24b",
        "two_gaps": "#e45756",
        "oblique_gap": "#b279a2",
    }
    pivot = units.pivot(index="geometry", columns="method", values="masked_score_rmse")
    for geometry, row in pivot.iterrows():
        error_axis.plot(
            method_x,
            [row[method] for method in method_order],
            "o-",
            color=colours[geometry],
            markersize=3.2,
            linewidth=0.9,
            label=titles[geometry],
        )
    error_axis.set_xticks(method_x, method_order, rotation=15)
    error_axis.set_ylabel("masked-sector score RMSE")
    error_axis.set_title("(g) geometry-level masked error", fontsize=7.5)
    error_axis.legend(
        frameon=False,
        fontsize=5.2,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )

    calibration_axis = figure.add_subplot(lower[0, 2])
    calibration_axis.plot((0.0, 1.0), (0.0, 1.0), "--", color="black", linewidth=0.7)
    styles = {
        "screened": ("#8f99a3", "o"),
        "passive": ("#2a7da8", "s"),
        "graph": ("#159f74", "D"),
    }
    pooled = reliability.loc[reliability["scope"] == "pooled_descriptive"]
    for method in method_order:
        subset = pooled.loc[
            (pooled["method"] == method) & (pooled["n"] > 0)
        ].sort_values("bin")
        colour, marker = styles[method]
        calibration_axis.plot(
            subset["mean_predicted"],
            subset["mean_truth"],
            marker=marker,
            markersize=3.0,
            linewidth=0.9,
            color=colour,
            label=method,
        )
    calibration_axis.set_xlim(0.0, 1.0)
    calibration_axis.set_ylim(0.0, 1.0)
    calibration_axis.set_aspect("equal", adjustable="box")
    calibration_axis.set_xlabel("reconstructed bin mean")
    calibration_axis.set_ylabel("hidden bin mean")
    calibration_axis.set_title("(h) calibration", fontsize=7.5)
    calibration_axis.legend(frameon=False, fontsize=6.0)

    figure.savefig(FIG / "fig4_sparse_reconstruction.pdf")
    figure.savefig(FIG / "fig4_sparse_reconstruction.png")
    plt.close(figure)


def make_sparse_figure(results: pd.DataFrame, sensitivity: pd.DataFrame, representative: dict[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.85), constrained_layout=True)
    for axis, key, title in (
        (axes[0, 0], "truth", r"(a) hidden score $u^\dagger$"),
        (axes[0, 2], "graph", "(c) graph reconstruction"),
    ):
        image = axis.imshow(
            representative[key].T,
            origin="lower",
            extent=(0, LENGTH, 0, LENGTH),
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        figure.colorbar(image, ax=axis, shrink=0.68, pad=0.02)

    confidence = representative["confidence"]
    plotted_confidence = np.ma.log10(np.ma.masked_less_equal(confidence, 0.0))
    axes[0, 1].set_facecolor("white")
    image = axes[0, 1].imshow(
        plotted_confidence.T,
        origin="lower",
        extent=(0, LENGTH, 0, LENGTH),
        cmap="viridis",
        aspect="equal",
    )
    axes[0, 1].contour(
        representative["holdout_mask"].T,
        levels=[0.5],
        colors="#c83e4d",
        linewidths=0.8,
        origin="lower",
        extent=(0, LENGTH, 0, LENGTH),
    )
    axes[0, 1].scatter(
        representative["train_x"],
        representative["train_y"],
        s=1.2,
        color="black",
        alpha=0.45,
        linewidths=0,
    )
    axes[0, 1].set_title(r"(b) compact confidence; white: $\lambda=0$")
    axes[0, 1].set_xlabel("x (mm)")
    axes[0, 1].set_ylabel("y (mm)")
    figure.colorbar(image, ax=axes[0, 1], shrink=0.68, pad=0.02, label=r"$\log_{10}\lambda$")

    order = ("screened", "passive", "graph")
    colours = ("#8f99a3", "#2a7da8", "#159f74")
    boxes = axes[1, 0].boxplot(
        [results.loc[results["method"] == method, "rmse"] for method in order],
        tick_labels=order,
        patch_artist=True,
        showfliers=False,
    )
    for box, colour in zip(boxes["boxes"], colours):
        box.set_facecolor(colour)
        box.set_alpha(0.8)
    paired = results.pivot(index="seed", columns="method", values="rmse")
    for _, row in paired.iterrows():
        axes[1, 0].plot(
            (1, 2, 3),
            [row[method] for method in order],
            color="#777777",
            alpha=0.22,
            linewidth=0.45,
            zorder=0,
        )
    axes[1, 0].set_title("(d) independent test-seed error")
    axes[1, 0].set_ylabel("field RMSE")
    axes[1, 0].tick_params(axis="x", rotation=18)

    boxes = axes[1, 1].boxplot(
        [results.loc[results["method"] == method, "capacity_abs_error"] for method in order],
        tick_labels=order,
        patch_artist=True,
        showfliers=False,
    )
    for box, colour in zip(boxes["boxes"], colours):
        box.set_facecolor(colour)
        box.set_alpha(0.8)
    paired = results.pivot(index="seed", columns="method", values="capacity_abs_error")
    for _, row in paired.iterrows():
        axes[1, 1].plot(
            (1, 2, 3),
            [row[method] for method in order],
            color="#777777",
            alpha=0.22,
            linewidth=0.45,
            zorder=0,
        )
    axes[1, 1].set_title("(e) effective-conductance error")
    axes[1, 1].set_ylabel(r"$|C_h^\ast-C_{h,\mathrm{true}}^\ast|$")
    axes[1, 1].tick_params(axis="x", rotation=18)

    for initialisation, colour, marker in (
        ("screened", "#159f74", "o"),
        ("constant_-0.8", "#e07a3f", "s"),
    ):
        subset = sensitivity.loc[sensitivity["initialisation"] == initialisation]
        axes[1, 2].plot(
            subset["horizon_multiplier"],
            subset["capacity_abs_error"],
            marker=marker,
            color=colour,
            label=initialisation.replace("_", " "),
        )
    axes[1, 2].set_xticks((0.5, 1.0, 2.0), (r"$T/2$", r"$T$", r"$2T$"))
    axes[1, 2].set_ylabel("conductance error")
    axes[1, 2].set_title("(f) graph: horizon and initialisation")
    axes[1, 2].legend(frameon=False, fontsize=6.5)

    figure.savefig(FIG / "fig4_sparse_reconstruction.pdf")
    figure.savefig(FIG / "fig4_sparse_reconstruction.png")
    plt.close(figure)


def make_capacity_figure(
    phase: pd.DataFrame,
    ep: pd.DataFrame,
    refinement: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        2, 3, figsize=(7.15, 5.35), constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 0.74)},
    )
    figure.get_layout_engine().set(rect=(0.0, 0.105, 1.0, 0.895))
    angles = sorted(phase["gap_angle_deg"].unique())
    capacity_norm = mpl.colors.LogNorm(
        vmin=float(phase["normalized_capacity"].min()),
        vmax=float(phase["normalized_capacity"].max()),
    )

    def cell_edges(values: np.ndarray, logarithmic: bool = False) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        transformed = np.log(values) if logarithmic else values
        middle = 0.5 * (transformed[:-1] + transformed[1:])
        edges = np.r_[
            transformed[0] - 0.5 * (transformed[1] - transformed[0]),
            middle,
            transformed[-1] + 0.5 * (transformed[-1] - transformed[-2]),
        ]
        return np.exp(edges) if logarithmic else edges

    for axis, angle_deg, panel in zip(axes[0], angles, ("a", "b", "c")):
        subset = phase.loc[np.isclose(phase["gap_angle_deg"], angle_deg)]
        pivot = subset.pivot(
            index="gap_diffusivity_fraction",
            columns="gap_width_mm",
            values="normalized_capacity",
        ).sort_index()
        widths = pivot.columns.to_numpy(dtype=float)
        diffusivities = pivot.index.to_numpy(dtype=float)
        image = axis.pcolormesh(
            cell_edges(widths),
            cell_edges(diffusivities, logarithmic=True),
            pivot.to_numpy(),
            shading="flat",
            cmap="viridis",
            norm=capacity_norm,
        )
        axis.set_yscale("log")
        axis.set_ylim(
            cell_edges(diffusivities, logarithmic=True)[0],
            cell_edges(diffusivities, logarithmic=True)[-1],
        )
        axis.set_title(rf"({panel}) $\theta={angle_deg:.0f}^\circ$")
        axis.set_xlabel("zero-level gap width (mm)")
        axis.set_ylabel("nominal gap diffusivity fraction")
    figure.colorbar(
        image,
        ax=axes[0, :].tolist(),
        shrink=0.78,
        pad=0.015,
        label=r"$C_h^\ast$",
    )

    styles = {"exit": "o", "entrance": "s"}
    angle_norm = mpl.colors.Normalize(vmin=0.0, vmax=90.0)
    angle_cmap = mpl.colormaps["plasma"]
    for direction, marker in styles.items():
        subset = ep.loc[
            (ep["experiment"] == "stratified")
            & (ep["pacing_direction"] == direction)
        ]
        finite = np.isfinite(subset["sector_first_arrival_ms"])
        axes[1, 0].scatter(
            subset.loc[finite, "normalized_capacity"],
            subset.loc[finite, "sector_first_arrival_ms"],
            c=subset.loc[finite, "gap_angle_deg"],
            cmap=angle_cmap,
            norm=angle_norm,
            marker=marker,
            s=23,
            edgecolors="none",
            label=direction,
        )
        censored = subset.loc[~finite]
        axes[1, 0].scatter(
            censored["normalized_capacity"],
            censored["t_end_ms"],
            facecolors="none",
            edgecolors=angle_cmap(angle_norm(censored["gap_angle_deg"])),
            marker=marker,
            s=25,
            linewidths=0.8,
        )
    horizon = float(ep["t_end_ms"].max())
    axes[1, 0].axhline(
        horizon,
        color="#666666",
        linestyle=":",
        linewidth=0.8,
        label="no capture by horizon",
    )
    complete_ring = ep.loc[
        (ep["experiment"] == "complete_ring_control")
        & (ep["pacing_direction"] == "exit")
    ].iloc[0]
    axes[1, 0].scatter(
        [complete_ring["normalized_capacity"]],
        [complete_ring["t_end_ms"]],
        marker="D",
        facecolors="none",
        edgecolors="black",
        s=27,
        linewidths=0.8,
        label="complete ring",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel(r"normalised conductance $C_h^\ast$")
    axes[1, 0].set_ylabel("sector first arrival (ms)")
    axes[1, 0].set_title("(d) bidirectional EP response")
    axes[1, 0].legend(
        frameon=False, fontsize=6.0, loc="upper left",
        bbox_to_anchor=(0.0, -0.29), ncol=2,
        borderaxespad=0.0, columnspacing=0.7, handletextpad=0.4,
    ).set_in_layout(False)
    figure.colorbar(
        mpl.cm.ScalarMappable(norm=angle_norm, cmap=angle_cmap),
        ax=axes[1, 0],
        shrink=0.65,
        pad=0.02,
        label="gap angle (deg)",
    )

    capacity_refinement = refinement.loc[
        refinement["quantity"] == "capacity"
    ].sort_values("dx_mm")
    finest = float(capacity_refinement.iloc[0]["normalized_capacity"])
    coarser = capacity_refinement.iloc[1:]
    axes[1, 1].semilogy(
        coarser["dx_mm"],
        np.abs(coarser["normalized_capacity"] - finest) / abs(finest),
        "o-",
        color="#2a7da8",
        label="representative vs finest",
    )
    matched_mesh = (
        refinement.loc[refinement["quantity"] == "matched_EP_space"]
        .groupby("n", as_index=False)
        .first()
        .sort_values("dx_mm")
    )
    axes[1, 1].semilogy(
        matched_mesh["dx_mm"],
        matched_mesh["matched_pair_relative_capacity_difference"],
        "s--",
        color="#e07a3f",
        label="matched-pair difference",
    )
    axes[1, 1].set_xticks(
        capacity_refinement["dx_mm"],
        [f"{value:.2f}" for value in capacity_refinement["dx_mm"]],
    )
    axes[1, 1].set_xlabel("cell width (mm)")
    axes[1, 1].set_ylabel("relative difference")
    axes[1, 1].set_title("(e) mesh sensitivity")
    axes[1, 1].legend(
        frameon=False, fontsize=5.7, loc="upper left",
        bbox_to_anchor=(0.0, -0.29), borderaxespad=0.0,
    ).set_in_layout(False)
    face_mean = ep.loc[ep["experiment"] == "face_mean_sensitivity"].set_index(
        "face_average"
    )
    face_change = 100.0 * (
        face_mean.loc["arithmetic", "normalized_capacity"]
        / face_mean.loc["harmonic", "normalized_capacity"]
        - 1.0
    )
    axes[1, 1].text(
        0.0,
        -0.48,
        f"arithmetic: $C_h^*$ +{face_change:.1f}%\nno capture for either mean",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=5.7,
    ).set_in_layout(False)

    matched = ep.loc[ep["experiment"] == "matched_capacity"].copy()
    case_order = ("A", "B")
    for direction, marker, colour, offset in (
        ("exit", "o", "#2a7da8", -0.035),
        ("entrance", "s", "#e07a3f", 0.035),
    ):
        subset = matched.loc[matched["pacing_direction"] == direction].set_index(
            "matched_case"
        )
        axes[1, 2].plot(
            np.arange(2) + offset,
            [subset.loc[case, "target_peak_voltage"] for case in case_order],
            marker=marker,
            color=colour,
            label=direction,
        )
    capacity_labels = []
    for case in case_order:
        row = matched.loc[matched["matched_case"] == case].iloc[0]
        capacity_labels.append(
            f"{case}\n$w={row['gap_width_mm']:g},\\eta={row['gap_diffusivity_fraction']:g},"
            f"\\theta={row['gap_angle_deg']:g}^\\circ$"
        )
    axes[1, 2].set_xticks(np.arange(2), capacity_labels)
    axes[1, 2].tick_params(axis="x", labelsize=5.8)
    axes[1, 2].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1, 2].set_ylabel("distal-sector peak voltage")
    axes[1, 2].set_title("(f) matched conductance")
    axes[1, 2].legend(frameon=False)

    figure.savefig(FIG / "fig5_pvi_capacity_phase_diagram.pdf")
    figure.savefig(FIG / "fig5_pvi_capacity_phase_diagram.png")
    plt.close(figure)


def make_end_to_end_figure(
    frame: pd.DataFrame,
    saved: dict[str, np.ndarray],
    ep_refinement: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 5.05), constrained_layout=True)
    for axis, key, title in (
        (axes[0, 0], "truth_diffusivity", "(a) hidden diffusivity"),
        (axes[0, 1], "graph_diffusivity", "(b) reconstructed diffusivity"),
    ):
        image = axis.imshow(
            saved[key].T,
            origin="lower",
            extent=(0, LENGTH, 0, LENGTH),
            norm=mpl.colors.LogNorm(vmin=ETA_MIN, vmax=1.0),
            cmap="viridis",
            aspect="equal",
        )
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_title(title)
    figure.colorbar(
        image,
        ax=[axes[0, 0], axes[0, 1]],
        shrink=0.68,
        pad=0.02,
        label="diffusivity fraction",
    )

    order = ("screened", "passive", "graph")
    colours = {"screened": "#8f99a3", "passive": "#2a7da8", "graph": "#159f74"}
    boxes = axes[0, 2].boxplot(
        [frame.loc[frame["method"] == method, "normalized_capacity"] for method in order],
        tick_labels=order,
        patch_artist=True,
        showfliers=False,
    )
    for box, method in zip(boxes["boxes"], order):
        box.set_facecolor(colours[method])
        box.set_alpha(0.8)
    truth_capacity = float(frame["truth_normalized_capacity"].iloc[0])
    axes[0, 2].axhline(
        truth_capacity, linestyle="--", color="black", linewidth=0.9
    )
    axes[0, 2].text(
        0.98,
        truth_capacity,
        " hidden truth",
        transform=axes[0, 2].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.2,
    )
    axes[0, 2].set_ylabel("normalised conductance")
    axes[0, 2].set_title("(c) conductance")
    axes[0, 2].tick_params(axis="x", rotation=18)

    paired_capacity = frame.pivot(
        index="seed", columns="method", values="capacity_abs_error"
    ).dropna()
    capacity_difference = (
        paired_capacity["graph"] - paired_capacity["passive"]
    ).to_numpy(dtype=float)
    rank = np.arange(1, len(capacity_difference) + 1)
    rng = np.random.default_rng(20260903)
    bootstrap_means = np.mean(
        rng.choice(
            capacity_difference,
            size=(5000, len(capacity_difference)),
            replace=True,
        ),
        axis=1,
    )
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].fill_between(
        (0.5, len(capacity_difference) + 0.5),
        lower,
        upper,
        color="#159f74",
        alpha=0.16,
        linewidth=0,
    )
    axes[1, 0].axhline(
        float(np.mean(capacity_difference)),
        color="#159f74",
        linewidth=1.0,
    )
    axes[1, 0].scatter(
        rank,
        capacity_difference,
        color="#159f74",
        s=13,
        zorder=3,
    )
    axes[1, 0].set_xlabel("paired test replicate rank")
    axes[1, 0].set_ylabel(r"$|\Delta C|_{\rm graph}-|\Delta C|_{\rm passive}$")
    axes[1, 0].set_title("(d) paired conductance gain")
    axes[1, 0].set_xlim(0.5, len(capacity_difference) + 0.5)
    axes[1, 0].set_xticks((1, 4, 8, 12, 16))

    ep_pair = frame.loc[frame["ep_included"] == 1].copy()
    seeds = sorted(ep_pair["seed"].unique())
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    direction_style = {
        "exit": ("o", "#2a7da8"),
        "entrance": ("s", "#e07a3f"),
    }
    for method, offset in (
        ("passive", -0.07),
        ("graph", 0.07),
    ):
        subset = ep_pair.loc[ep_pair["method"] == method].sort_values("seed")
        x = np.asarray([seed_index[seed] for seed in subset["seed"]], dtype=float) + offset
        for direction, (marker, colour) in direction_style.items():
            axes[1, 1].scatter(
                x,
                subset[f"{direction}_crossing_time_ms"],
                marker=marker,
                facecolors="white" if method == "passive" else colour,
                edgecolors=colour,
                s=16,
                linewidths=0.8,
                label=f"{method}, {direction}",
            )
    for direction, (_, colour) in direction_style.items():
        axes[1, 1].axhline(
            float(frame[f"truth_{direction}_crossing_time_ms"].iloc[0]),
            color=colour,
            linewidth=0.8,
            alpha=0.55,
            linestyle=":",
        )
    axes[1, 1].set_xlabel("independent test seed")
    axes[1, 1].set_ylabel("crossing time (ms)")
    axes[1, 1].set_title("(e) bidirectional ensemble")
    axes[1, 1].legend(
        frameon=False,
        fontsize=5.0,
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.50, 0.47),
        handletextpad=0.3,
        columnspacing=0.6,
    )

    refinement = ep_refinement.sort_values("dx_mm", ascending=False)
    axes[1, 2].plot(
        refinement["dx_mm"],
        refinement["crossing_time_ms"],
        "o-",
        color="#2a7da8",
        markersize=4,
    )
    axes[1, 2].set_xlabel("cell width (mm)")
    axes[1, 2].set_ylabel("exit first-arrival time (ms)")
    capacity_variation = 100.0 * (
        ep_refinement["normalized_capacity"].max()
        - ep_refinement["normalized_capacity"].min()
    ) / ep_refinement["normalized_capacity"].mean()
    axes[1, 2].text(
        0.04,
        0.96,
        f"$C_h^*$ range: {capacity_variation:.2f}%",
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=6.0,
    )
    axes[1, 2].set_title("(f) EP mesh sensitivity")

    figure.savefig(FIG / "fig6_end_to_end_uncertainty.pdf")
    figure.savefig(FIG / "fig6_end_to_end_uncertainty.png")
    plt.close(figure)


def save_current_applied_outputs(
    geometry_outputs: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, np.ndarray],
    ],
    phase: pd.DataFrame,
    ep: pd.DataFrame,
    refinement: pd.DataFrame,
    capacity_representatives: dict[str, np.ndarray],
) -> None:
    """Write only the current Experiment 4 and Experiment 5 artifacts."""

    save_geometry_reconstruction_outputs(*geometry_outputs)
    phase.to_csv(DATA / "pvi_capacity_phase_diagram.csv", index=False)
    ep.to_csv(DATA / "pvi_bidirectional_ep.csv", index=False)
    refinement.to_csv(DATA / "pvi_capacity_refinement.csv", index=False)
    np.savez_compressed(
        DATA / "pvi_capacity_representatives.npz", **capacity_representatives
    )


# RETIRED LEGACY PATH: this writer is intentionally not called by ``main`` or
# exposed by the CLI.  It is kept only to document the provenance of old
# single-gap and synthetic end-to-end files that may remain in an unpacked
# working directory.
def _save_retired_legacy_outputs(
    horizon_selection: pd.DataFrame,
    sparse: pd.DataFrame,
    sensitivity: pd.DataFrame,
    common_horizon: pd.DataFrame,
    reconstruction_refinement: pd.DataFrame,
    representative: dict[str, np.ndarray],
    phase: pd.DataFrame,
    ep: pd.DataFrame,
    refinement: pd.DataFrame,
    capacity_representatives: dict[str, np.ndarray],
    end_to_end: pd.DataFrame,
    summary: pd.DataFrame,
    end_to_end_ep_refinement: pd.DataFrame,
    end_to_end_saved: dict[str, np.ndarray],
) -> None:
    horizon_selection.to_csv(DATA / "reconstruction_horizon_selection.csv", index=False)
    sparse.to_csv(DATA / "sparse_reconstruction_ensemble.csv", index=False)
    sensitivity.to_csv(DATA / "sparse_reconstruction_sensitivity.csv", index=False)
    common_horizon.to_csv(
        DATA / "sparse_reconstruction_common_horizon.csv", index=False
    )
    summarize_common_horizon_ablation(common_horizon).to_csv(
        DATA / "sparse_reconstruction_common_horizon_summary.csv", index=False
    )
    reconstruction_refinement.to_csv(
        DATA / "sparse_reconstruction_refinement.csv", index=False
    )
    np.savez_compressed(DATA / "sparse_reconstruction_maps.npz", **representative)
    phase.to_csv(DATA / "pvi_capacity_phase_diagram.csv", index=False)
    ep.to_csv(DATA / "pvi_bidirectional_ep.csv", index=False)
    refinement.to_csv(DATA / "pvi_capacity_refinement.csv", index=False)
    np.savez_compressed(
        DATA / "pvi_capacity_representatives.npz", **capacity_representatives
    )
    end_to_end.to_csv(DATA / "end_to_end_ensemble.csv", index=False)
    summary.to_csv(DATA / "end_to_end_summary.csv", index=False)
    end_to_end_ep_refinement.to_csv(
        DATA / "end_to_end_ep_refinement.csv", index=False
    )
    np.savez_compressed(DATA / "end_to_end_fields.npz", **end_to_end_saved)


def write_run_metadata(mode: str = "publication") -> None:
    """Record the runtime, fixed design, and checksums of the reproducibility set."""

    tracked = [Path(relative) for relative in PUBLICATION_ARTIFACT_PATHS]
    missing = [relative.as_posix() for relative in tracked if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot write publication metadata; missing declared artifacts: "
            + ", ".join(missing)
        )
    artifacts = {}
    for relative in tracked:
        payload = (ROOT / relative).read_bytes()
        artifacts[relative.as_posix()] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    metadata = {
        "schema_version": 1,
        "mode": mode,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in ("numpy", "scipy", "pandas", "matplotlib")
            },
        },
        "commands": [
            "python code/run_core.py",
            "python code/run_applied.py",
            "python code/run_applied.py --geometry-reconstruction-only",
            "python code/run_applied.py --matched-refinement-only",
            "python code/run_zenodo_pvi_reconstruction.py --cohort-root /path/to/zenodo_erp/meshes",
            "python code/check_patient_contact_geometry.py --cohort-root /path/to/zenodo_erp/meshes",
            "python code/run_applied.py --refresh-run-metadata",
            "python -m unittest discover -s code -p 'test_*.py' -v",
            "python code/validate_outputs.py",
        ],
        "design": {
            "publication_experiments": [1, 2, 3, 4, 5, 6],
            "patient_experiment": "zenodo_prior_pvi_surface_reconstruction",
            "patient_source_doi": "10.5281/zenodo.10726677",
            "patient_replication_unit": "patient_after_averaging_left_right_masks",
            "patient_ids": ["P1", "P3", "P4", "P5", "P6", "P7"],
            "geometry_reconstruction_horizon": GEOMETRY_RECONSTRUCTION_HORIZON,
            "geometry_horizon_rule": "fixed_common_120_step_budget_no_selection",
            "geometry_blackout_angles_deg": list(GEOMETRY_BLACKOUT_ANGLES_DEG),
            "geometry_acquisition_seeds": list(GEOMETRY_ACQUISITION_SEEDS),
            "geometry_replication_unit": "geometry",
            "geometry_contact_noise": "uncensored_gaussian",
            "geometry_contact_noise_sd": CONTACT_NOISE_SD,
            "geometry_names": [
                str(specification["geometry"]) for specification in GEOMETRY_SPECS
            ],
            "reconstruction_n": RECON_N,
            "reconstruction_dt": PHASE_DT,
            "terminal_window_states": TERMINAL_WINDOW,
            "terminal_window_pseudo_time": TERMINAL_WINDOW * PHASE_DT,
            "production_ep_n": PRODUCTION_EP_N,
            "production_ep_dt_ms": PRODUCTION_EP_DT,
            "matched_pair_spatial_grids": [81, 121, 161],
            "matched_pair_time_steps_ms": [0.01, 0.02, 0.04],
            "matched_pair_pacing_directions": ["exit", "entrance"],
            "subcells_per_axis": SUBCELLS_PER_AXIS,
            "finite_volume_face_average": "harmonic",
            "target_capture_fraction": TARGET_CAPTURE_FRACTION,
        },
        "artifacts": artifacts,
    }
    (DATA / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def refresh_run_metadata_from_saved_outputs() -> None:
    """Refresh checksums for the explicit current publication artifact list."""

    write_run_metadata(mode="publication")
    print("publication run metadata refreshed from the explicit artifact list")


def smoke_test() -> None:
    grid = reconstruction_grid(33)
    blackout_angle = np.deg2rad(GEOMETRY_BLACKOUT_ANGLES_DEG[0])
    truth_for_contacts = geometry_score_field(grid.X, grid.Y, GEOMETRY_SPECS[1]["gaps"])
    contacts = _build_geometry_contacts(
        grid, GEOMETRY_ACQUISITION_SEEDS[0], truth_for_contacts, blackout_angle
    )
    confidence, data_forcing = compact_observation_fields(
        grid, contacts, blackout_angle=blackout_angle
    )
    smoke_holdout = holdout_mask(grid, blackout_angle=blackout_angle)
    if float(np.max(confidence[smoke_holdout])) != 0.0:
        raise AssertionError("held-out confidence is not exactly zero")
    if float(np.max(np.abs(data_forcing[smoke_holdout]))) != 0.0:
        raise AssertionError("held-out data forcing is not exactly zero")
    width_grid = reconstruction_grid(101)
    truth = lesion_score_field(width_grid.X, width_grid.Y)
    realised_width = ring_gap_width(
        truth,
        width_grid.X,
        width_grid.Y,
        CENTER,
        RADIUS,
        gap_angle=TRUTH_GAP_ANGLE,
    )
    if abs(realised_width - TRUTH_GAP_WIDTH) > 0.05:
        raise AssertionError("the prescribed zero-level gap width was not realised")
    initial, screen = screened_initial_state(grid, confidence, data_forcing)
    if screen["screen_relative_residual"] > 1.0e-8:
        raise AssertionError("screened initial-state residual is too large")
    state, history = run_reconstruction_history(
        grid, initial, confidence, data_forcing, "graph", 12
    )
    field = terminal_average(history, 12, window=6)
    if not np.isfinite(field).all():
        raise AssertionError("reconstruction contains non-finite values")
    scale = analytic_subcell_diffusivity(41, 6.0, 0.03, np.deg2rad(45.0), subcells=2)
    capacity, _ = capacity_metrics(scale)
    if capacity["capacity_flux_energy_defect"] > 1.0e-8:
        raise AssertionError("capacity flux and energy do not agree")
    if max(item["state_residual"] for item in state.diagnostics) > 6.0e-7:
        raise AssertionError("graph solve residual is too large")
    geometry_truth = geometry_score_field(
        grid.X, grid.Y, ((4.0, 0.0, 1.4), (6.0, 180.0, 0.0))
    )
    geometry_ring = _ring_gap_geometry(grid, geometry_truth)
    if geometry_ring["gap_count"] != 2:
        raise AssertionError("multi-gap geometry does not retain two components")
    calibration = _continuous_calibration_metrics(geometry_truth, geometry_truth)
    if calibration["scaled_score_mse"] > 1.0e-25:
        raise AssertionError("identity continuous-score calibration is inconsistent")
    ep_result, _ = ep_readout(scale, np.deg2rad(45.0), "exit", t_end=3.0)
    if ep_result["target_cell_count"] <= 0:
        raise AssertionError("the distal EP target sector is empty")
    print("strengthened applied-suite smoke test passed")


def main(quick: bool = False) -> None:
    """Run the current synthetic publication experiments (Experiments 4 and 5).

    Superseded single-gap horizon selection and the synthetic end-to-end
    ensemble remain as provenance code above, but are deliberately unreachable
    from this publication entry point.
    """

    geometry_outputs = run_geometry_replicated_reconstruction(quick=quick)
    phase, ep, refinement, capacity_representatives = run_capacity_phase_diagram(quick)
    print(
        "bidirectional EP counts\n",
        ep.groupby(["pacing_direction", "gap_angle_deg"])["crossed_by_horizon"].agg(
            ["count", "sum"]
        ),
    )

    save_current_applied_outputs(
        geometry_outputs,
        phase,
        ep,
        refinement,
        capacity_representatives,
    )
    make_geometry_reconstruction_figure(
        geometry_outputs[1], geometry_outputs[3], geometry_outputs[4]
    )
    make_capacity_figure(phase, ep, refinement)
    print(
        "Experiments 4 and 5 written. After the separate patient-surface run, "
        "refresh the complete publication checksums with "
        "`python code/run_applied.py --refresh-run-metadata`."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run short structural checks without writing publication outputs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="write a reduced development dataset; default runs the publication suite",
    )
    parser.add_argument(
        "--geometry-reconstruction-only",
        action="store_true",
        help="recompute only the five-geometry fixed-block reconstruction experiment",
    )
    parser.add_argument(
        "--matched-refinement-only",
        action="store_true",
        help="recompute only the matched-capacity spatial/time refinement from saved publication data",
    )
    parser.add_argument(
        "--refresh-run-metadata",
        action="store_true",
        help="refresh publication checksums after all targeted regenerations are complete",
    )
    arguments = parser.parse_args()
    selected_modes = sum(
        (
            arguments.smoke_test,
            arguments.quick,
            arguments.geometry_reconstruction_only,
            arguments.matched_refinement_only,
            arguments.refresh_run_metadata,
        )
    )
    if selected_modes > 1:
        parser.error("all execution-mode flags are mutually exclusive")
    if arguments.smoke_test:
        smoke_test()
    elif arguments.geometry_reconstruction_only:
        refresh_geometry_reconstruction()
    elif arguments.matched_refinement_only:
        refresh_matched_capacity_refinement()
    elif arguments.refresh_run_metadata:
        refresh_run_metadata_from_saved_outputs()
    else:
        main(quick=arguments.quick)
