"""Check the graph predictor's primal--dual algebraic error estimate.

These are fixed, independently declared algebraic checks on a Fourier chart
and a curved P1 surface. They do not modify Experiments 1, 2, or 5 and do not
measure spatial or temporal discretisation error. The residual form avoids
subtracting two nearly equal objective values near convergence.
"""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from model import PhaseParameters, PhaseState, SpectralGrid, phase_step
from surface_fem import assemble_p1
from surface_mesh import icosphere_tri_mesh
from surface_phase import (
    SurfacePhaseParameters,
    assemble_frozen_surface_system,
    frozen_surface_predictor,
)


ROOT = Path(__file__).resolve().parents[1]


def certificate(c, p, cap, rhs, apply_l, apply_b, solve_b, weights):
    """Return the computable gap and the B-norm bound, in weighted units."""
    feasible = np.clip(p, -cap, cap)
    lc = apply_l(c)
    residual = apply_b(c) - rhs + apply_l(feasible)
    inverse_residual = solve_b(residual)
    stationarity_term = float(np.sum(weights * residual * inverse_residual))
    complementarity = float(
        np.sum(weights * np.maximum(cap * np.abs(lc) - feasible * lc, 0.0))
    )
    if stationarity_term < -1.0e-24:
        raise AssertionError("negative B-inverse quadratic form")
    gap = 0.5 * max(stationarity_term, 0.0) + complementarity
    shifted_rhs = rhs - apply_l(feasible)
    primal = (
        0.5 * np.sum(weights * c * apply_b(c))
        - np.sum(weights * rhs * c)
        + np.sum(weights * cap * np.abs(lc))
    )
    dual = -0.5 * np.sum(weights * shifted_rhs * solve_b(shifted_rhs))
    identity_defect = float(abs((primal - dual) - gap))
    if identity_defect > 1.0e-11 * max(1.0, abs(primal), abs(dual)):
        raise AssertionError("primal--dual gap identity failed")
    return {
        "gap": gap,
        "B_error_bound": float(np.sqrt(2.0 * gap)),
        "rms_error_bound": float(np.sqrt(2.0 * gap / np.sum(weights))),
        "stationarity_Binverse_squared": stationarity_term,
        "complementarity_defect": complementarity,
        "gap_identity_absolute_defect": identity_defect,
    }


def run_backend(name, solve, cap, rhs, apply_l, apply_b, solve_b, weights):
    reference, multiplier, _ = solve(1.0e-11)
    reference_certificate = certificate(
        reference, multiplier, cap, rhs, apply_l, apply_b, solve_b, weights
    )
    rows = []
    for tolerance in (1.0e-4, 1.0e-6, 1.0e-8):
        state, multiplier, iterations = solve(tolerance)
        values = certificate(state, multiplier, cap, rhs, apply_l, apply_b, solve_b, weights)
        difference = state - reference
        difference_b = float(np.sqrt(max(np.sum(weights * difference * apply_b(difference)), 0.0)))
        difference_rms = float(np.sqrt(np.sum(weights * difference**2) / np.sum(weights)))
        # Both computed solutions have a bound relative to the same minimiser.
        if difference_b > values["B_error_bound"] + reference_certificate["B_error_bound"] + 1.0e-12:
            raise AssertionError("observed predictor difference exceeds both bounds")
        rows.append({
            "backend": name,
            "degrees_of_freedom": state.size,
            "admm_tolerance": tolerance,
            "admm_iterations": iterations,
            **values,
            "B_difference_from_tight_solve": difference_b,
            "rms_difference_from_tight_solve": difference_rms,
            "reference_rms_error_bound": reference_certificate["rms_error_bound"],
        })
    return rows


def main():
    grid = SpectralGrid(33, 33, kx_weight=1.0, ky_weight=0.65)
    initial = (
        0.42 * np.cos(2.0 * np.pi * grid.X) * np.cos(2.0 * np.pi * grid.Y)
        + 0.13 * np.sin(4.0 * np.pi * grid.X)
    )
    phase = PhaseParameters(mu=0.003, nu=0.002, dt=5.0e-5, classifier="graph")
    alpha = phase.dt * phase.nu
    symbol_b = 1.0 + phase.dt * 4.0 * grid.A + phase.dt * phase.mu * grid.A2
    nonlinear = grid.dealiased_cubic(initial) - initial
    rhs = grid.ifft(
        (1.0 + phase.dt * 4.0 * grid.A) * grid.fft(initial)
        - phase.dt * grid.A * grid.fft(nonlinear)
    )
    cap = alpha * grid.grad_norm_k(initial)
    zeros = np.zeros_like(initial)

    def spectral_solve(tolerance):
        result = phase_step(
            grid, PhaseState(initial.copy()), zeros, zeros,
            replace(phase, admm_tol=tolerance),
        )
        return result.u, phase.rho_factor * alpha * result.y, result.diagnostics[-1]["admm_iterations"]

    rows = run_backend(
        "Fourier_33x33", spectral_solve, cap, rhs,
        grid.anisotropic_laplacian,
        lambda value: grid.ifft(symbol_b * grid.fft(value)),
        lambda value: grid.ifft(grid.fft(value) / symbol_b),
        np.full_like(initial, 1.0 / initial.size),
    )

    mesh = icosphere_tri_mesh(2)
    operators = assemble_p1(mesh)
    x, y, z = mesh.points.T
    current = 0.35 * x + 0.15 * y * z
    parameters = SurfacePhaseParameters.graph(mu=0.03, nu=0.02, dt=0.003)
    system = assemble_frozen_surface_system(operators, current, parameters)
    mass = operators.vertex_area

    def surface_solve(tolerance):
        result = frozen_surface_predictor(
            operators, current, replace(parameters, admm_tolerance=tolerance)
        )
        return result.predictor, result.multiplier, result.diagnostics["admm_iterations"]

    rows += run_backend(
        "P1_sphere_162", surface_solve,
        system.alpha * system.gradient_factor, system.rhs_strong,
        operators.apply_laplacian,
        lambda value: (system.matrix @ value) / mass,
        lambda value: spsolve(system.matrix, mass * value),
        mass,
    )
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    with (directory / "graph_error_certificate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "graph_error_certificate_protocol.json").write_text(json.dumps({
        "purpose": "fixed-grid primal-dual algebraic error check; no discretisation claim",
        "Fourier": {"nx": 33, "ny": 33, "mu": 0.003, "nu": 0.002, "dt": 5.0e-5, "K_diagonal": [1.0, 0.65]},
        "surface": {"mesh": "unit icosphere, subdivision level 2", "vertices": mesh.n_vertices, "mu": 0.03, "nu": 0.02, "dt": 0.003},
        "tolerances": [1.0e-4, 1.0e-6, 1.0e-8],
        "tight_comparison_tolerance": 1.0e-11,
        "normalisation": "rms columns divide mass norm by sqrt(total quadrature weight)",
        "finite_precision": "bounds are evaluated in float64; no interval-arithmetic rounding certificate is claimed",
    }, indent=2) + "\n")
    for row in rows:
        print(f"{row['backend']} tol={row['admm_tolerance']:.0e} "
              f"bound={row['rms_error_bound']:.3e} "
              f"difference={row['rms_difference_from_tight_solve']:.3e}")


if __name__ == "__main__":
    main()
