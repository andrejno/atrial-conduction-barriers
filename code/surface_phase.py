"""Mesh-native screened, passive, and sign-graph reconstruction kernels.

Let ``D`` be the lumped P1 mass matrix, ``K`` the positive surface stiffness
matrix, and ``L = -D^{-1} K`` the non-positive discrete Laplace--Beltrami
operator.  The implementation is the mass-weighted counterpart of the
manuscript's fixed-grid convex predictor:

    D B = D + dt S K + dt mu K D^{-1} K.

For the direct graph predictor, weighted ADMM solves the constraint ``z=Lc``.
The stationarity residual in weak coordinates is ``DB c - D r - K p``, where
``p=rho*y``.  All reported residuals are recomputed from the accepted state;
an iteration-limit or linear-solver failure raises instead of returning the
last iterate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg, splu

try:  # Supports direct scripts and package-style imports.
    from .surface_fem import SurfaceP1Operators
except ImportError:  # pragma: no cover
    from surface_fem import SurfaceP1Operators


FloatArray = NDArray[np.float64]
LinearMethod = Literal["direct", "cg"]


@dataclass(frozen=True)
class SurfacePhaseParameters:
    """Parameters for one frozen-coefficient surface predictor."""

    mu: float = 0.30
    nu: float = 0.20
    dt: float = 0.03
    classifier: Literal["passive", "graph"] = "graph"
    rho_factor: float = 20.0
    admm_tolerance: float = 5.0e-8
    admm_max_iterations: int = 8000
    linear_method: LinearMethod = "direct"
    linear_tolerance: float = 1.0e-11
    linear_absolute_tolerance: float = 1.0e-13
    linear_max_iterations: int = 4000
    enforce_data_compatibility: bool = True

    @classmethod
    def passive(cls, **kwargs: object) -> "SurfacePhaseParameters":
        """Construct a passive parameter set without a silently ignored ``nu``."""

        return cls(classifier="passive", nu=0.0, **kwargs)

    @classmethod
    def graph(cls, **kwargs: object) -> "SurfacePhaseParameters":
        return cls(classifier="graph", **kwargs)

    def validate(self) -> None:
        finite_positive = {
            "mu": self.mu,
            "dt": self.dt,
            "rho_factor": self.rho_factor,
            "admm_tolerance": self.admm_tolerance,
            "linear_tolerance": self.linear_tolerance,
            "linear_absolute_tolerance": self.linear_absolute_tolerance,
        }
        for name, value in finite_positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if not np.isfinite(self.nu) or self.nu < 0.0:
            raise ValueError("nu must be finite and non-negative")
        if self.classifier not in {"passive", "graph"}:
            raise ValueError("classifier must be 'passive' or 'graph'")
        if self.classifier == "passive" and self.nu != 0.0:
            raise ValueError("passive parameters require nu=0")
        if self.classifier == "graph" and self.nu <= 0.0:
            raise ValueError("graph parameters require nu>0")
        if self.admm_max_iterations < 1 or self.linear_max_iterations < 1:
            raise ValueError("iteration limits must be positive integers")
        if self.linear_method not in {"direct", "cg"}:
            raise ValueError("linear_method must be 'direct' or 'cg'")


@dataclass(frozen=True)
class FrozenSurfaceSystem:
    """Mass-weighted quadratic predictor system with frozen explicit terms."""

    matrix: csr_matrix
    biharmonic: csr_matrix
    rhs_weak: FloatArray
    rhs_strong: FloatArray
    gradient_factor: FloatArray
    stabilisation: float
    alpha: float


@dataclass(frozen=True)
class SurfaceScreenedResult:
    state: FloatArray
    weak_residual: FloatArray
    strong_residual: FloatArray
    iterations: int
    relative_residual: float
    objective: float


@dataclass(frozen=True)
class SurfacePredictorResult:
    predictor: FloatArray
    laplacian: FloatArray
    z: FloatArray
    scaled_dual: FloatArray
    multiplier: FloatArray
    gradient_factor: FloatArray
    weak_residual: FloatArray
    strong_residual: FloatArray
    diagnostics: dict[str, float]


@dataclass
class SurfacePhaseState:
    u: FloatArray
    z: FloatArray | None = None
    scaled_dual: FloatArray | None = None
    predictor: FloatArray | None = None
    multiplier: FloatArray | None = None
    diagnostics: list[dict[str, float]] = field(default_factory=list)
    # Only linear algebra is cached.  A factor is reused after exact equality
    # of the sparse matrix and all solver controls has been checked.
    _linear_solver_cache: dict[str, _SPDLinearSolver] = field(
        default_factory=dict, repr=False
    )


def _vector(values: ArrayLike, size: int, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.asarray(vector, dtype=float)


def _validate_operators(operators: SurfaceP1Operators) -> FloatArray:
    size = operators.n_vertices
    mass = np.asarray(operators.vertex_area, dtype=float)
    if mass.shape != (size,) or not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise ValueError("lumped vertex masses must be finite and strictly positive")
    for name, matrix in (
        ("lumped_mass", operators.lumped_mass),
        ("stiffness", operators.stiffness),
        ("unit_stiffness", operators.unit_stiffness),
    ):
        if matrix.shape != (size, size):
            raise ValueError(f"{name} has an incompatible shape")
        if not np.isfinite(matrix.data).all():
            raise ValueError(f"{name} contains a non-finite entry")
    diagonal = np.asarray(operators.lumped_mass.diagonal(), dtype=float)
    scale = max(float(np.max(mass)), 1.0)
    if float(np.max(np.abs(diagonal - mass))) > 1.0e-12 * scale:
        raise ValueError("lumped_mass diagonal and vertex_area are inconsistent")
    stiffness_scale = max(float(np.sqrt(np.sum(operators.stiffness.data**2))), 1.0)
    asymmetry = operators.stiffness - operators.stiffness.T
    if float(np.sqrt(np.sum(asymmetry.data**2))) > 1.0e-12 * stiffness_scale:
        raise ValueError("stiffness matrix is not symmetric")
    return mass


def _mass_norm(values: FloatArray, mass: FloatArray) -> float:
    return float(np.sqrt(max(float(np.dot(mass, values * values)), 0.0)))


def _relative_mass_norm(
    residual: FloatArray,
    mass: FloatArray,
    *references: FloatArray,
) -> float:
    denominator = 1.0
    for reference in references:
        denominator = max(denominator, _mass_norm(reference, mass))
    return float(_mass_norm(residual, mass) / denominator)


def _check_observations(
    confidence: ArrayLike,
    data_forcing: ArrayLike,
    size: int,
    *,
    enforce_compatibility: bool,
) -> tuple[FloatArray, FloatArray]:
    lam = _vector(confidence, size, "confidence")
    forcing = _vector(data_forcing, size, "data_forcing")
    if np.any(lam < 0.0):
        raise ValueError("confidence must be non-negative; exact zeros are permitted")
    if enforce_compatibility:
        tolerance = 1.0e-12 * max(1.0, float(np.max(lam)))
        violation = np.abs(forcing) - lam
        if float(np.max(violation)) > tolerance:
            raise ValueError("data_forcing violates |f| <= confidence")
        if np.any((lam == 0.0) & (forcing != 0.0)):
            raise ValueError("data_forcing must vanish wherever confidence is zero")
    return lam, forcing


class _SPDLinearSolver:
    def __init__(
        self,
        matrix: csr_matrix,
        *,
        method: LinearMethod,
        tolerance: float,
        absolute_tolerance: float,
        max_iterations: int,
        label: str,
    ) -> None:
        self.matrix = matrix.tocsr()
        self.method = method
        self.tolerance = tolerance
        self.absolute_tolerance = absolute_tolerance
        self.max_iterations = max_iterations
        self.label = label
        diagonal = np.asarray(self.matrix.diagonal(), dtype=float)
        if not np.isfinite(self.matrix.data).all() or np.any(diagonal <= 0.0):
            raise ValueError(f"{label} is not a finite positive-diagonal system")
        self._factor = None
        self._direct_scale = None
        self._extended_matrix = None
        if method == "direct":
            # Symmetric equilibration changes coordinates, not the operator:
            # (S A S) y = S b, x = S y.  It limits the dynamic range in the
            # sparse factors on surfaces with very unequal vertex masses.
            self._direct_scale = 1.0 / np.sqrt(diagonal)
            scale = diags(self._direct_scale, format="csr")
            try:
                self._factor = splu((scale @ self.matrix @ scale).tocsc())
            except RuntimeError as exc:
                raise np.linalg.LinAlgError(f"could not factor {label}: {exc}") from exc

    def _direct_solve(self, rhs: FloatArray) -> FloatArray:
        assert self._factor is not None and self._direct_scale is not None
        return np.asarray(
            self._direct_scale * self._factor.solve(self._direct_scale * rhs),
            dtype=float,
        )

    def solve(
        self,
        rhs: FloatArray,
        initial: FloatArray | None = None,
    ) -> tuple[FloatArray, int, float]:
        if not np.isfinite(rhs).all():
            raise ValueError(f"{self.label} right-hand side contains a non-finite value")
        iterations = 1
        if self.method == "direct":
            solution = self._direct_solve(rhs)
        else:
            inverse_diagonal = 1.0 / np.asarray(self.matrix.diagonal(), dtype=float)
            preconditioner = LinearOperator(
                self.matrix.shape,
                matvec=lambda value: inverse_diagonal * value,
                dtype=float,
            )
            iterations = 0

            def callback(_: FloatArray) -> None:
                nonlocal iterations
                iterations += 1

            solution, info = cg(
                self.matrix,
                rhs,
                x0=initial,
                M=preconditioner,
                rtol=self.tolerance,
                atol=self.absolute_tolerance,
                maxiter=self.max_iterations,
                callback=callback,
            )
            if info != 0:
                detail = (
                    f"iteration limit {info}" if info > 0 else f"illegal input/breakdown {info}"
                )
                raise RuntimeError(f"{self.label} CG failed: {detail}")
            solution = np.asarray(solution, dtype=float)
        if not np.isfinite(solution).all():
            raise FloatingPointError(f"{self.label} solve returned a non-finite state")
        residual = np.asarray(self.matrix @ solution - rhs, dtype=float)
        denominator = max(float(np.linalg.norm(rhs)), 1.0)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        acceptance = max(50.0 * self.tolerance, 5.0e-12)
        if self.method == "direct" and relative_residual > acceptance:
            # Defect correction solves A delta = b-Ax with the same factors.
            # Forming the defect in extended precision reduces cancellation
            # in the biharmonic term.  Acceptance is still checked with the
            # original float64 matrix and exactly the same residual bound.
            if self._extended_matrix is None:
                self._extended_matrix = self.matrix.astype(np.longdouble)
            candidate = solution.copy()
            rhs_extended = rhs.astype(np.longdouble)
            for _ in range(4):
                defect = np.asarray(
                    rhs_extended
                    - self._extended_matrix @ candidate.astype(np.longdouble),
                    dtype=float,
                )
                candidate = candidate + self._direct_solve(defect)
                iterations += 1
                if not np.isfinite(candidate).all():
                    break
                candidate_residual = np.asarray(self.matrix @ candidate - rhs)
                candidate_relative = float(
                    np.linalg.norm(candidate_residual) / denominator
                )
                if candidate_relative < relative_residual:
                    solution = candidate.copy()
                    relative_residual = candidate_relative
                if relative_residual <= acceptance:
                    break
        if relative_residual > acceptance:
            raise RuntimeError(
                f"{self.label} residual {relative_residual:.3e} exceeds {acceptance:.3e}"
            )
        return solution, iterations, relative_residual


def vertex_gradient_factor(
    operators: SurfaceP1Operators,
    values: ArrayLike,
) -> FloatArray:
    """Area-lumped nodal average of ``sqrt(a) |grad_S u|``.

    For a scalar reconstruction metric ``a``, this is the P1 counterpart of
    ``sqrt(<K grad u, grad u>)``.  It reproduces a constant affine gradient on
    a planar mesh exactly.
    """

    mass = _validate_operators(operators)
    u = _vector(values, operators.n_vertices, "values")
    triangles = operators.mesh.triangles
    points = operators.mesh.points[triangles]
    edge_1 = points[:, 1] - points[:, 0]
    edge_2 = points[:, 2] - points[:, 0]
    g00 = np.einsum("ij,ij->i", edge_1, edge_1)
    g01 = np.einsum("ij,ij->i", edge_1, edge_2)
    g11 = np.einsum("ij,ij->i", edge_2, edge_2)
    determinant = g00 * g11 - g01 * g01
    if np.any(determinant <= 0.0):
        raise ValueError("a triangle has a singular intrinsic metric")
    du1 = u[triangles[:, 1]] - u[triangles[:, 0]]
    du2 = u[triangles[:, 2]] - u[triangles[:, 0]]
    gradient_squared = (
        g11 * du1 * du1 - 2.0 * g01 * du1 * du2 + g00 * du2 * du2
    ) / determinant
    metric_gradient = np.sqrt(
        np.maximum(operators.cell_coefficient * gradient_squared, 0.0)
    )
    accumulator = np.zeros(operators.n_vertices, dtype=float)
    contributions = operators.cell_area * metric_gradient / 3.0
    for local_index in range(3):
        np.add.at(accumulator, triangles[:, local_index], contributions)
    result = accumulator / mass
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise FloatingPointError("computed gradient factor is invalid")
    return result


def screened_reconstruction(
    operators: SurfaceP1Operators,
    confidence: ArrayLike,
    data_forcing: ArrayLike,
    *,
    epsilon: float = 0.05,
    length_scale: float = 1.5,
    linear_method: LinearMethod = "direct",
    linear_tolerance: float = 1.0e-11,
    linear_absolute_tolerance: float = 1.0e-13,
    linear_max_iterations: int = 4000,
    enforce_data_compatibility: bool = True,
) -> SurfaceScreenedResult:
    """Solve ``(epsilon+lambda)c - ell^2 Lc = f`` in lumped weak form."""

    mass = _validate_operators(operators)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and strictly positive")
    for name, value in (
        ("linear_tolerance", linear_tolerance),
        ("linear_absolute_tolerance", linear_absolute_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if linear_max_iterations < 1:
        raise ValueError("linear_max_iterations must be positive")
    if linear_method not in {"direct", "cg"}:
        raise ValueError("linear_method must be 'direct' or 'cg'")
    lam, forcing = _check_observations(
        confidence,
        data_forcing,
        operators.n_vertices,
        enforce_compatibility=enforce_data_compatibility,
    )
    matrix = (
        diags(mass * (epsilon + lam), format="csr")
        + length_scale**2 * operators.stiffness
    ).tocsr()
    rhs = mass * forcing
    solver = _SPDLinearSolver(
        matrix,
        method=linear_method,
        tolerance=linear_tolerance,
        absolute_tolerance=linear_absolute_tolerance,
        max_iterations=linear_max_iterations,
        label="screened reconstruction",
    )
    state, iterations, linear_residual = solver.solve(rhs)
    weak_residual = np.asarray(matrix @ state - rhs, dtype=float)
    strong_residual = weak_residual / mass
    relative_residual = _relative_mass_norm(strong_residual, mass, forcing)
    acceptance = max(100.0 * linear_tolerance, 1.0e-10)
    if relative_residual > acceptance:
        raise RuntimeError(
            f"screened strong residual {relative_residual:.3e} exceeds {acceptance:.3e}"
        )
    objective = float(0.5 * state @ (matrix @ state) - rhs @ state)
    return SurfaceScreenedResult(
        state=state,
        weak_residual=weak_residual,
        strong_residual=strong_residual,
        iterations=iterations,
        relative_residual=max(relative_residual, linear_residual),
        objective=objective,
    )


def assemble_frozen_surface_system(
    operators: SurfaceP1Operators,
    state: ArrayLike,
    parameters: SurfacePhaseParameters,
    *,
    external_forcing: ArrayLike | None = None,
    gradient_factor: ArrayLike | None = None,
) -> FrozenSurfaceSystem:
    """Assemble the mass-weighted quadratic part and explicit right-hand side."""

    parameters.validate()
    mass = _validate_operators(operators)
    u = _vector(state, operators.n_vertices, "state")
    if external_forcing is None:
        forcing = np.zeros(operators.n_vertices, dtype=float)
    else:
        forcing = _vector(external_forcing, operators.n_vertices, "external_forcing")
    if gradient_factor is None:
        gradient = vertex_gradient_factor(operators, u)
    else:
        gradient = _vector(gradient_factor, operators.n_vertices, "gradient_factor")
        if np.any(gradient < 0.0):
            raise ValueError("gradient_factor must be non-negative")

    maximum = float(np.max(np.abs(u)))
    stabilisation = max(4.0, 1.0 + 3.0 * max(1.0, maximum) ** 2)
    biharmonic = operators.lumped_biharmonic_matrix()
    matrix = (
        operators.lumped_mass
        + parameters.dt * stabilisation * operators.stiffness
        + parameters.dt * parameters.mu * biharmonic
    ).tocsr()
    double_well_derivative = u**3 - u
    rhs_weak = np.asarray(
        mass * u
        + parameters.dt * stabilisation * (operators.stiffness @ u)
        - parameters.dt * (operators.stiffness @ double_well_derivative)
        + parameters.dt * mass * forcing,
        dtype=float,
    )
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs_weak).all():
        raise FloatingPointError("frozen predictor system contains a non-finite value")
    return FrozenSurfaceSystem(
        matrix=matrix,
        biharmonic=biharmonic,
        rhs_weak=rhs_weak,
        rhs_strong=rhs_weak / mass,
        gradient_factor=gradient,
        stabilisation=stabilisation,
        alpha=parameters.dt * parameters.nu,
    )


def _shrink(values: FloatArray, threshold: FloatArray) -> FloatArray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def frozen_surface_predictor(
    operators: SurfaceP1Operators,
    state: ArrayLike,
    parameters: SurfacePhaseParameters,
    *,
    external_forcing: ArrayLike | None = None,
    gradient_factor: ArrayLike | None = None,
    z0: ArrayLike | None = None,
    scaled_dual0: ArrayLike | None = None,
    _linear_solver_cache: dict[str, _SPDLinearSolver] | None = None,
) -> SurfacePredictorResult:
    """Solve one passive or direct-sign-graph frozen predictor exactly to tolerance."""

    system = assemble_frozen_surface_system(
        operators,
        state,
        parameters,
        external_forcing=external_forcing,
        gradient_factor=gradient_factor,
    )
    mass = _validate_operators(operators)
    size = operators.n_vertices
    linear_iterations = 0
    linear_max_residual = 0.0
    factorizations = 0
    solver_reuses = 0

    def linear_solver(matrix: csr_matrix, label: str) -> _SPDLinearSolver:
        nonlocal factorizations, solver_reuses
        candidate = None if _linear_solver_cache is None else _linear_solver_cache.get(label)
        if (
            candidate is not None
            and candidate.method == parameters.linear_method
            and candidate.tolerance == parameters.linear_tolerance
            and candidate.absolute_tolerance == parameters.linear_absolute_tolerance
            and candidate.max_iterations == parameters.linear_max_iterations
            and candidate.matrix.shape == matrix.shape
            and np.array_equal(candidate.matrix.indptr, matrix.indptr)
            and np.array_equal(candidate.matrix.indices, matrix.indices)
            and np.array_equal(candidate.matrix.data, matrix.data)
        ):
            solver_reuses += 1
            return candidate
        solver = _SPDLinearSolver(
            matrix,
            method=parameters.linear_method,
            tolerance=parameters.linear_tolerance,
            absolute_tolerance=parameters.linear_absolute_tolerance,
            max_iterations=parameters.linear_max_iterations,
            label=label,
        )
        factorizations += int(parameters.linear_method == "direct")
        if _linear_solver_cache is not None:
            # There are three fixed labels, with at most one factor per label.
            _linear_solver_cache[label] = solver
        return solver

    if parameters.classifier == "passive":
        solver = linear_solver(system.matrix, "passive surface predictor")
        predictor, count, linear_residual = solver.solve(system.rhs_weak)
        linear_iterations += count
        linear_max_residual = linear_residual
        laplacian = np.asarray(-(operators.stiffness @ predictor) / mass, dtype=float)
        z = laplacian.copy()
        scaled_dual = np.zeros(size, dtype=float)
        multiplier = np.zeros(size, dtype=float)
        primal_relative = dual_relative = 0.0
        iterations = 0
    else:
        alpha = system.alpha
        rho = parameters.rho_factor * alpha
        augmented_matrix = (
            system.matrix + rho * system.biharmonic
        ).tocsr()
        solver = linear_solver(augmented_matrix, "graph ADMM predictor")
        if z0 is None:
            passive_solver = linear_solver(system.matrix, "graph initial passive predictor")
            predictor, count, linear_residual = passive_solver.solve(system.rhs_weak)
            linear_iterations += count
            linear_max_residual = max(linear_max_residual, linear_residual)
            z = np.asarray(-(operators.stiffness @ predictor) / mass, dtype=float)
        else:
            z = _vector(z0, size, "z0").copy()
            predictor = np.asarray(state, dtype=float).copy()
        if scaled_dual0 is None:
            scaled_dual = np.zeros(size, dtype=float)
        else:
            scaled_dual = _vector(scaled_dual0, size, "scaled_dual0").copy()

        primal_relative = dual_relative = float("inf")
        for iterations in range(1, parameters.admm_max_iterations + 1):
            rhs = system.rhs_weak - rho * (
                operators.stiffness @ (z - scaled_dual)
            )
            predictor, count, linear_residual = solver.solve(rhs, initial=predictor)
            linear_iterations += count
            linear_max_residual = max(linear_max_residual, linear_residual)
            laplacian = np.asarray(-(operators.stiffness @ predictor) / mass, dtype=float)
            previous_z = z
            z = _shrink(
                laplacian + scaled_dual,
                alpha * system.gradient_factor / rho,
            )
            scaled_dual = scaled_dual + laplacian - z
            primal = laplacian - z
            dual = np.asarray(
                -rho * (operators.stiffness @ (z - previous_z)) / mass,
                dtype=float,
            )
            primal_relative = _relative_mass_norm(primal, mass, laplacian)
            dual_relative = _relative_mass_norm(dual, mass, system.rhs_strong)
            if max(primal_relative, dual_relative) <= parameters.admm_tolerance:
                break
        else:
            raise RuntimeError(
                "surface graph ADMM failed after "
                f"{parameters.admm_max_iterations} iterations: "
                f"primal={primal_relative:.3e}, dual={dual_relative:.3e}"
            )
        multiplier = rho * scaled_dual

    weak_residual = np.asarray(
        system.matrix @ predictor
        - system.rhs_weak
        - operators.stiffness @ multiplier,
        dtype=float,
    )
    strong_residual = weak_residual / mass
    quadratic_strong = np.asarray((system.matrix @ predictor) / mass, dtype=float)
    state_relative = _relative_mass_norm(
        strong_residual,
        mass,
        quadratic_strong,
        system.rhs_strong,
    )
    cap = system.alpha * system.gradient_factor
    box_violation = float(np.max(np.maximum(np.abs(multiplier) - cap, 0.0)))
    complementarity_numerator = float(
        np.dot(mass, np.abs(cap * np.abs(laplacian) - multiplier * laplacian))
    )
    complementarity_denominator = max(
        float(np.dot(mass, cap * np.abs(laplacian))), 1.0e-30
    )
    complementarity_relative = complementarity_numerator / complementarity_denominator
    gamma = 1.0 / max(1.0, _mass_norm(laplacian, mass))
    projected = np.clip(multiplier + gamma * laplacian, -cap, cap)
    projection_residual = multiplier - projected
    projection_relative = _mass_norm(projection_residual, mass) / max(
        _mass_norm(cap, mass), 1.0
    )
    objective = float(
        0.5 * predictor @ (system.matrix @ predictor)
        - system.rhs_weak @ predictor
        + system.alpha
        * np.dot(mass * system.gradient_factor, np.abs(laplacian))
    )
    residual_acceptance = max(
        20.0 * parameters.admm_tolerance,
        100.0 * parameters.linear_tolerance,
        1.0e-9,
    )
    if state_relative > residual_acceptance:
        raise RuntimeError(
            f"accepted predictor state residual {state_relative:.3e} exceeds "
            f"{residual_acceptance:.3e}"
        )
    if parameters.classifier == "graph":
        box_scale = max(float(np.max(cap)), 1.0)
        if box_violation > residual_acceptance * box_scale:
            raise RuntimeError("accepted graph multiplier violates its pointwise box")
        if projection_relative > residual_acceptance:
            raise RuntimeError(
                f"accepted graph projection residual {projection_relative:.3e} exceeds "
                f"{residual_acceptance:.3e}"
            )

    diagnostics = {
        "admm_iterations": float(iterations),
        "linear_iterations": float(linear_iterations),
        "linear_factorizations": float(factorizations),
        "linear_solver_reuses": float(solver_reuses),
        "linear_max_relative_residual": float(linear_max_residual),
        "primal_relative_residual": float(primal_relative),
        "dual_relative_residual": float(dual_relative),
        "state_relative_residual": float(state_relative),
        "weak_residual_l2": float(np.linalg.norm(weak_residual)),
        "strong_residual_mass_norm": _mass_norm(strong_residual, mass),
        "box_violation_max": box_violation,
        "complementarity_relative_residual": float(complementarity_relative),
        "graph_projection_relative_residual": float(projection_relative),
        "objective": objective,
        "stabilisation": float(system.stabilisation),
        "gradient_factor_min": float(np.min(system.gradient_factor)),
        "gradient_factor_max": float(np.max(system.gradient_factor)),
        "predictor_min": float(np.min(predictor)),
        "predictor_max": float(np.max(predictor)),
    }
    return SurfacePredictorResult(
        predictor=predictor,
        laplacian=laplacian,
        z=z,
        scaled_dual=scaled_dual,
        multiplier=multiplier,
        gradient_factor=system.gradient_factor,
        weak_residual=weak_residual,
        strong_residual=strong_residual,
        diagnostics=diagnostics,
    )


def surface_phase_step(
    operators: SurfaceP1Operators,
    state: SurfacePhaseState,
    confidence: ArrayLike,
    data_forcing: ArrayLike,
    parameters: SurfacePhaseParameters,
    *,
    external_forcing: ArrayLike | None = None,
    gradient_factor: ArrayLike | None = None,
) -> SurfacePhaseState:
    """Advance one predictor/fidelity split step on the triangular surface."""

    parameters.validate()
    mass = _validate_operators(operators)
    u = _vector(state.u, operators.n_vertices, "state.u")
    lam, forcing = _check_observations(
        confidence,
        data_forcing,
        operators.n_vertices,
        enforce_compatibility=parameters.enforce_data_compatibility,
    )
    if external_forcing is None:
        external = np.zeros(operators.n_vertices, dtype=float)
    else:
        external = _vector(external_forcing, operators.n_vertices, "external_forcing")
    predictor = frozen_surface_predictor(
        operators,
        u,
        parameters,
        external_forcing=external,
        gradient_factor=gradient_factor,
        _linear_solver_cache=state._linear_solver_cache,
        z0=state.z if parameters.classifier == "graph" else None,
        scaled_dual0=state.scaled_dual if parameters.classifier == "graph" else None,
    )
    denominator = 1.0 + parameters.dt * lam
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise ValueError("implicit fidelity denominator is not strictly positive")
    updated = (predictor.predictor + parameters.dt * forcing) / denominator
    if not np.isfinite(updated).all():
        raise FloatingPointError("fidelity step returned a non-finite state")
    source = forcing - lam * updated + external
    mass_defect = abs(float(np.dot(mass, updated - u - parameters.dt * source))) / float(
        np.sum(mass)
    )
    predictor_mass_defect = abs(
        float(np.dot(mass, predictor.predictor - u - parameters.dt * external))
    ) / float(np.sum(mass))
    diagnostics = dict(predictor.diagnostics)
    diagnostics.update(
        {
            "mass_source_defect": mass_defect,
            "predictor_mass_defect": predictor_mass_defect,
            "confidence_min": float(np.min(lam)),
            "confidence_max": float(np.max(lam)),
            "zero_confidence_fraction": float(np.mean(lam == 0.0)),
            "state_min": float(np.min(updated)),
            "state_max": float(np.max(updated)),
        }
    )
    defect_acceptance = max(
        100.0 * parameters.linear_tolerance,
        20.0 * parameters.admm_tolerance,
        1.0e-10,
    )
    if mass_defect > defect_acceptance or predictor_mass_defect > defect_acceptance:
        raise RuntimeError(
            "surface phase mass identity failed: "
            f"full={mass_defect:.3e}, predictor={predictor_mass_defect:.3e}"
        )
    return SurfacePhaseState(
        u=updated,
        z=predictor.z.copy(),
        scaled_dual=predictor.scaled_dual.copy(),
        predictor=predictor.predictor.copy(),
        multiplier=predictor.multiplier.copy(),
        diagnostics=state.diagnostics + [diagnostics],
        _linear_solver_cache=state._linear_solver_cache,
    )


def solve_surface_phase(
    operators: SurfaceP1Operators,
    initial: ArrayLike,
    confidence: ArrayLike,
    data_forcing: ArrayLike,
    parameters: SurfacePhaseParameters,
    nsteps: int,
    *,
    external_forcing_callback: Callable[[int, float, FloatArray], ArrayLike] | None = None,
    keep_history: bool = False,
) -> tuple[SurfacePhaseState, list[FloatArray] | None]:
    """Run a deterministic finite pseudo-time trajectory."""

    if not isinstance(nsteps, (int, np.integer)) or nsteps < 0:
        raise ValueError("nsteps must be a non-negative integer")
    parameters.validate()
    _validate_operators(operators)
    state = SurfacePhaseState(
        u=_vector(initial, operators.n_vertices, "initial").copy()
    )
    history: list[FloatArray] | None = [state.u.copy()] if keep_history else None
    for step in range(nsteps):
        external = (
            None
            if external_forcing_callback is None
            else external_forcing_callback(
                step + 1, (step + 1) * parameters.dt, state.u.copy()
            )
        )
        state = surface_phase_step(
            operators,
            state,
            confidence,
            data_forcing,
            parameters,
            external_forcing=external,
        )
        if history is not None:
            history.append(state.u.copy())
    return state, history


__all__ = [
    "FrozenSurfaceSystem",
    "SurfacePhaseParameters",
    "SurfacePhaseState",
    "SurfacePredictorResult",
    "SurfaceScreenedResult",
    "assemble_frozen_surface_system",
    "frozen_surface_predictor",
    "screened_reconstruction",
    "solve_surface_phase",
    "surface_phase_step",
    "vertex_gradient_factor",
]
