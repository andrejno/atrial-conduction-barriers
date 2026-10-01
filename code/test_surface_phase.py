"""Manufactured algebra and failure-mode tests for surface reconstruction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.fft import dst, idst
from scipy.sparse import diags, eye


CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from surface_fem import assemble_p1
from surface_mesh import rectangular_tri_mesh
from surface_phase import (
    _SPDLinearSolver,
    SurfacePhaseParameters,
    SurfacePhaseState,
    assemble_frozen_surface_system,
    frozen_surface_predictor,
    screened_reconstruction,
    solve_surface_phase,
    surface_phase_step,
    vertex_gradient_factor,
)


class SurfacePhaseManufacturedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = rectangular_tri_mesh(7, 5, length=1.4, width=1.0)
        self.operators = assemble_p1(self.mesh)
        self.mass = self.operators.vertex_area
        self.x = self.mesh.points[:, 0]
        self.y = self.mesh.points[:, 1]
        self.current = 0.20 * np.sin(1.2 * self.x) + 0.15 * np.cos(2.1 * self.y)
        self.target = (
            0.35 * np.sin(np.pi * self.x / 1.4) * np.cos(np.pi * self.y)
            + 0.08 * self.x
        )

    def test_screened_manufactured_state_and_residual(self) -> None:
        epsilon, length_scale = 0.07, 0.4
        confidence = 0.2 + 0.3 * self.x
        matrix = (
            diags(self.mass * (epsilon + confidence), format="csr")
            + length_scale**2 * self.operators.stiffness
        )
        forcing = np.asarray(matrix @ self.target) / self.mass
        result = screened_reconstruction(
            self.operators,
            confidence,
            forcing,
            epsilon=epsilon,
            length_scale=length_scale,
            enforce_data_compatibility=False,
        )
        error = self.operators.mass_norm(result.state - self.target)
        self.assertLess(error, 2.0e-14)
        np.testing.assert_allclose(
            result.weak_residual,
            matrix @ result.state - self.mass * forcing,
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            result.strong_residual,
            result.weak_residual / self.mass,
            rtol=0.0,
            atol=1.0e-15,
        )
        self.assertLess(result.relative_residual, 2.0e-14)

    def test_iterative_and_direct_linear_solvers_agree(self) -> None:
        confidence = 0.5 + 0.2 * self.x
        forcing = 0.3 * confidence * np.sin(self.x)
        direct = screened_reconstruction(
            self.operators,
            confidence,
            forcing,
            epsilon=0.1,
            length_scale=0.4,
        )
        iterative = screened_reconstruction(
            self.operators,
            confidence,
            forcing,
            epsilon=0.1,
            length_scale=0.4,
            linear_method="cg",
            linear_tolerance=1.0e-12,
            linear_max_iterations=1000,
        )
        self.assertGreater(iterative.iterations, 1)
        self.assertLess(
            self.operators.mass_norm(iterative.state - direct.state), 2.0e-12
        )
        self.assertLess(iterative.relative_residual, 2.0e-11)

    def test_factor_reuse_preserves_states_and_invalidates_changed_system(self) -> None:
        parameters = SurfacePhaseParameters.graph(
            dt=0.02, mu=0.03, nu=0.01, admm_tolerance=1.0e-9
        )
        cache = {}
        factor_counts = []
        for current in (self.current, self.current + 0.02, self.current + 1.4):
            cached = frozen_surface_predictor(
                self.operators, current, parameters, _linear_solver_cache=cache
            )
            uncached = frozen_surface_predictor(self.operators, current, parameters)
            np.testing.assert_array_equal(cached.predictor, uncached.predictor)
            np.testing.assert_array_equal(cached.multiplier, uncached.multiplier)
            factor_counts.append(cached.diagnostics["linear_factorizations"])
        self.assertEqual(factor_counts, [2.0, 0.0, 2.0])
        # A changed tolerance also invalidates the cached solver controls.
        tightened = frozen_surface_predictor(
            self.operators,
            self.current + 1.4,
            replace(parameters, linear_tolerance=1.0e-12),
            _linear_solver_cache=cache,
        )
        self.assertEqual(tightened.diagnostics["linear_factorizations"], 2.0)
        self.assertEqual(len(cache), 2)

    def test_ill_conditioned_biharmonic_direct_solve(self) -> None:
        # A positive biharmonic system with condition number about 1e8.
        # Its independent discrete sine eigenbasis supplies a reference;
        # the small eigenvalues make factorization error observable.
        size = 160
        laplace = diags(
            (-np.ones(size - 1), 2.0 * np.ones(size), -np.ones(size - 1)),
            (-1, 0, 1),
            format="csr",
        )
        matrix = laplace @ laplace + 1.0e-8 * eye(size, format="csr")
        rhs = np.cos(2.0 * np.arange(size) / size)
        eigenvalues = 4.0 * np.sin(
            np.pi * np.arange(1, size + 1) / (2.0 * (size + 1))
        ) ** 2
        reference = idst(
            dst(rhs, type=1, norm="ortho") / (eigenvalues**2 + 1.0e-8),
            type=1,
            norm="ortho",
        )
        solver = _SPDLinearSolver(
            matrix,
            method="direct",
            tolerance=5.0e-11,
            absolute_tolerance=1.0e-13,
            max_iterations=1000,
            label="ill-conditioned biharmonic system",
        )
        solution, iterations, residual = solver.solve(rhs)
        recomputed = np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs)
        self.assertLessEqual(iterations, 5)
        self.assertLessEqual(residual, 2.5e-9)
        self.assertEqual(residual, recomputed)
        self.assertLess(
            np.linalg.norm(solution - reference) / np.linalg.norm(reference), 1.0e-8
        )

        # Refinement must not turn an unattainable residual request into
        # acceptance by replacing the original-coordinate stopping rule.
        strict_solver = _SPDLinearSolver(
            matrix,
            method="direct",
            tolerance=1.0e-15,
            absolute_tolerance=1.0e-15,
            max_iterations=1000,
            label="strict biharmonic system",
        )
        with self.assertRaisesRegex(RuntimeError, "residual .* exceeds"):
            strict_solver.solve(rhs)

    def test_passive_manufactured_predictor(self) -> None:
        parameters = SurfacePhaseParameters.passive(dt=0.04, mu=0.18)
        base = assemble_frozen_surface_system(
            self.operators, self.current, parameters
        )
        target_rhs = np.asarray(base.matrix @ self.target)
        external = (target_rhs - base.rhs_weak) / (parameters.dt * self.mass)
        result = frozen_surface_predictor(
            self.operators,
            self.current,
            parameters,
            external_forcing=external,
        )
        self.assertLess(
            self.operators.mass_norm(result.predictor - self.target), 2.0e-14
        )
        expected_residual = base.matrix @ result.predictor - target_rhs
        np.testing.assert_allclose(result.weak_residual, expected_residual, atol=2.0e-15)
        self.assertLess(result.diagnostics["state_relative_residual"], 5.0e-14)
        self.assertEqual(result.diagnostics["admm_iterations"], 0.0)

    def test_direct_graph_manufactured_kkt_system(self) -> None:
        parameters = SurfacePhaseParameters.graph(
            dt=0.04,
            mu=0.18,
            nu=0.22,
            rho_factor=30.0,
            admm_tolerance=1.0e-10,
            admm_max_iterations=20000,
        )
        gradient = 0.4 + 0.2 * np.sin(0.7 * self.x) ** 2 + 0.1 * self.y
        base = assemble_frozen_surface_system(
            self.operators,
            self.current,
            parameters,
            gradient_factor=gradient,
        )
        laplacian = -(self.operators.stiffness @ self.target) / self.mass
        capacity = base.alpha * gradient
        exact_multiplier = capacity * np.sign(laplacian)
        exact_multiplier[np.abs(laplacian) < 1.0e-12] = 0.0
        target_rhs = (
            base.matrix @ self.target
            - self.operators.stiffness @ exact_multiplier
        )
        external = (target_rhs - base.rhs_weak) / (parameters.dt * self.mass)
        result = frozen_surface_predictor(
            self.operators,
            self.current,
            parameters,
            external_forcing=external,
            gradient_factor=gradient,
        )
        normalized_error = self.operators.mass_norm(
            result.predictor - self.target
        ) / np.sqrt(self.operators.total_area)
        self.assertLess(normalized_error, 2.0e-11)
        self.assertLess(
            self.operators.mass_norm(result.multiplier - exact_multiplier), 2.0e-12
        )
        exact_weak_residual = (
            base.matrix @ result.predictor
            - target_rhs
            - self.operators.stiffness @ result.multiplier
        )
        np.testing.assert_allclose(
            result.weak_residual, exact_weak_residual, rtol=0.0, atol=2.0e-15
        )
        diagnostics = result.diagnostics
        self.assertLess(diagnostics["primal_relative_residual"], 1.0e-10)
        self.assertLess(diagnostics["dual_relative_residual"], 1.0e-10)
        self.assertLess(diagnostics["state_relative_residual"], 1.1e-10)
        self.assertLess(diagnostics["box_violation_max"], 1.0e-13)
        self.assertLess(diagnostics["complementarity_relative_residual"], 1.0e-12)
        self.assertLess(diagnostics["graph_projection_relative_residual"], 1.0e-12)

    def test_affine_gradient_factor_is_exact_and_rotation_invariant(self) -> None:
        mesh = rectangular_tri_mesh(8, 6, length=1.6, width=1.2)
        operators = assemble_p1(mesh, 4.0)
        values = 1.3 * mesh.points[:, 0] - 0.7 * mesh.points[:, 1] + 0.2
        computed = vertex_gradient_factor(operators, values)
        expected = 2.0 * np.sqrt(1.3**2 + 0.7**2)
        np.testing.assert_allclose(computed, expected, rtol=2.0e-14, atol=2.0e-14)

    def test_finite_trajectory_is_deterministic_and_closes_mass_balance(self) -> None:
        mesh = rectangular_tri_mesh(8, 5, length=1.6, width=1.0)
        operators = assemble_p1(mesh)
        x, y = mesh.points[:, 0], mesh.points[:, 1]
        confidence = np.where((x < 0.7) | (x > 1.2), 1.5, 0.0)
        score = np.tanh(4.0 * (0.18 - np.abs(y - 0.5)))
        forcing = confidence * score
        screened = screened_reconstruction(
            operators,
            confidence,
            forcing,
            epsilon=0.1,
            length_scale=0.25,
        )
        parameters = SurfacePhaseParameters.graph(
            dt=0.005,
            mu=0.08,
            nu=0.04,
            rho_factor=30.0,
            admm_tolerance=1.0e-8,
            admm_max_iterations=5000,
        )
        first, first_history = solve_surface_phase(
            operators,
            screened.state,
            confidence,
            forcing,
            parameters,
            4,
            keep_history=True,
        )
        second, second_history = solve_surface_phase(
            operators,
            screened.state,
            confidence,
            forcing,
            parameters,
            4,
            keep_history=True,
        )
        self.assertIsNotNone(first_history)
        self.assertIsNotNone(second_history)
        np.testing.assert_array_equal(first.u, second.u)
        for left, right in zip(first_history or [], second_history or []):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(len(first.diagnostics), 4)
        for diagnostic in first.diagnostics:
            self.assertLess(diagnostic["mass_source_defect"], 1.0e-12)
            self.assertLess(diagnostic["predictor_mass_defect"], 1.0e-12)
            self.assertGreater(diagnostic["zero_confidence_fraction"], 0.0)


class SurfacePhaseFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = rectangular_tri_mesh(5, 4)
        self.operators = assemble_p1(self.mesh)
        self.zeros = np.zeros(self.mesh.n_vertices)

    def test_negative_confidence_and_incompatible_forcing_fail(self) -> None:
        negative = self.zeros.copy()
        negative[0] = -1.0e-9
        with self.assertRaisesRegex(ValueError, "non-negative"):
            screened_reconstruction(self.operators, negative, self.zeros)
        incompatible = self.zeros.copy()
        incompatible[1] = 0.1
        with self.assertRaisesRegex(ValueError, r"\|f\| <= confidence"):
            screened_reconstruction(self.operators, self.zeros, incompatible)

    def test_nonfinite_state_and_nonpositive_mass_fail(self) -> None:
        bad_state = self.zeros.copy()
        bad_state[2] = np.nan
        parameters = SurfacePhaseParameters.passive(dt=0.01, mu=0.1)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            frozen_surface_predictor(self.operators, bad_state, parameters)
        bad_mass = self.operators.vertex_area.copy()
        bad_mass[0] = 0.0
        invalid_operators = replace(self.operators, vertex_area=bad_mass)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            frozen_surface_predictor(invalid_operators, self.zeros, parameters)

    def test_graph_iteration_limit_raises(self) -> None:
        x, y = self.mesh.points[:, 0], self.mesh.points[:, 1]
        state = np.sin(1.7 * x) + 0.3 * np.cos(2.2 * y)
        parameters = SurfacePhaseParameters.graph(
            dt=0.02,
            mu=0.1,
            nu=0.08,
            admm_tolerance=1.0e-14,
            admm_max_iterations=1,
        )
        with self.assertRaisesRegex(RuntimeError, "ADMM failed"):
            frozen_surface_predictor(self.operators, state, parameters)

    def test_fidelity_step_accepts_exactly_zero_confidence(self) -> None:
        parameters = SurfacePhaseParameters.passive(dt=0.01, mu=0.1)
        result = surface_phase_step(
            self.operators,
            SurfacePhaseState(self.zeros.copy()),
            self.zeros,
            self.zeros,
            parameters,
        )
        np.testing.assert_array_equal(result.u, self.zeros)
        self.assertEqual(result.diagnostics[-1]["zero_confidence_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
