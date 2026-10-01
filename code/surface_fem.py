"""Linear P1 finite elements and Dirichlet capacity on triangle surfaces.

The stiffness matrix represents

    K_ij = integral_S a grad_S(phi_i) . grad_S(phi_j) dS,

for a strictly positive isotropic scalar coefficient ``a``.  The assembly is
intrinsic: it uses the metric of each embedded triangle and is unchanged by a
rigid motion or a reversal of element orientation.  Natural zero-flux
conditions apply on surface boundaries not included in a Dirichlet node set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, spsolve

try:  # Supports both ``python code/file.py`` and package-style imports.
    from .surface_mesh import MeshQualityReport, TriangleMesh
except ImportError:  # pragma: no cover - exercised by the command-line scripts.
    from surface_mesh import MeshQualityReport, TriangleMesh


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class SurfaceP1Operators:
    """Mass, stiffness, and mass-weighted operators on one surface mesh."""

    mesh: TriangleMesh
    mass: csr_matrix
    lumped_mass: csr_matrix
    stiffness: csr_matrix
    unit_stiffness: csr_matrix
    vertex_area: FloatArray
    cell_area: FloatArray
    cell_coefficient: FloatArray
    quality: MeshQualityReport

    @property
    def n_vertices(self) -> int:
        return self.mesh.n_vertices

    @property
    def total_area(self) -> float:
        return float(np.sum(self.cell_area))

    def mass_inner(self, left: ArrayLike, right: ArrayLike) -> float:
        """Consistent-mass inner product ``left.T M right``."""

        u = _vertex_vector(left, self.n_vertices, "left")
        v = _vertex_vector(right, self.n_vertices, "right")
        return float(u @ (self.mass @ v))

    def mass_norm(self, values: ArrayLike) -> float:
        value = self.mass_inner(values, values)
        if value < -1.0e-13:
            raise FloatingPointError("mass quadratic form is unexpectedly negative")
        return float(np.sqrt(max(value, 0.0)))

    def mass_mean(self, values: ArrayLike) -> float:
        u = _vertex_vector(values, self.n_vertices, "values")
        return float(np.dot(self.vertex_area, u) / self.total_area)

    def apply_laplacian(self, values: ArrayLike, *, lumped: bool = True) -> FloatArray:
        """Apply the discrete Laplace--Beltrami operator ``M^{-1}(-K)``.

        The lumped form is local and is the default for nonlinear iterations.
        The consistent form solves one sparse mass system and is useful for
        verification.  Both annihilate constants to roundoff.
        """

        u = _vertex_vector(values, self.n_vertices, "values")
        rhs = -(self.stiffness @ u)
        if lumped:
            return np.asarray(rhs / self.vertex_area, dtype=float)
        return _safe_sparse_solve(self.mass, rhs, "consistent mass matrix")

    def laplacian_linear_operator(self, *, lumped: bool = True) -> LinearOperator:
        """Matrix-free strong Laplacian, self-adjoint in the M inner product."""

        def matvec(values: NDArray[np.float64]) -> FloatArray:
            return self.apply_laplacian(values, lumped=lumped)

        return LinearOperator(
            (self.n_vertices, self.n_vertices), matvec=matvec, dtype=float
        )

    def lumped_biharmonic_matrix(self) -> csr_matrix:
        """Return ``K M_lumped^{-1} K``, a symmetric weak biharmonic matrix."""

        inverse_mass = diags(1.0 / self.vertex_area, format="csr")
        matrix = self.stiffness @ inverse_mass @ self.stiffness
        return _symmetrise(matrix.tocsr())


@dataclass(frozen=True)
class SurfaceCapacityResult:
    """Capacity and algebraic identities for a unit-potential Dirichlet solve."""

    value: float
    normalized_value: float
    reference_capacity: float
    source_flux: float
    sink_flux: float
    energy: float
    relative_residual: float
    flux_energy_defect: float
    potential: FloatArray
    source_nodes: IntArray
    sink_nodes: IntArray


def _vertex_vector(values: ArrayLike, size: int, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains a non-finite value")
    return vector


def _symmetrise(matrix: csr_matrix) -> csr_matrix:
    symmetric = (0.5 * (matrix + matrix.T)).tocsr()
    symmetric.eliminate_zeros()
    return symmetric


def _coefficient_per_cell(
    mesh: TriangleMesh,
    coefficient: float | ArrayLike,
    coefficient_location: str,
) -> FloatArray:
    if coefficient_location not in {"auto", "cell", "node"}:
        raise ValueError("coefficient_location must be 'auto', 'cell', or 'node'")
    values = np.asarray(coefficient, dtype=float)
    if values.ndim == 0:
        cell_values = np.full(mesh.n_triangles, float(values), dtype=float)
    elif values.ndim == 1:
        if coefficient_location == "auto":
            is_cell = values.size == mesh.n_triangles
            is_node = values.size == mesh.n_vertices
            if is_cell and is_node:
                raise ValueError(
                    "coefficient length is ambiguous because n_vertices == n_triangles; "
                    "set coefficient_location explicitly"
                )
            if not (is_cell or is_node):
                raise ValueError(
                    "coefficient must be scalar or have one value per cell or vertex"
                )
            coefficient_location = "cell" if is_cell else "node"
        expected = mesh.n_triangles if coefficient_location == "cell" else mesh.n_vertices
        if values.size != expected:
            raise ValueError(
                f"{coefficient_location} coefficient has length {values.size}, expected {expected}"
            )
        if coefficient_location == "cell":
            cell_values = values.copy()
        else:
            # The integral of a P1 coefficient over a triangle is its nodal
            # mean times the area; gradients of the basis are cellwise constant.
            cell_values = np.mean(values[mesh.triangles], axis=1)
    else:
        raise ValueError("coefficient must be a scalar or one-dimensional array")
    if not np.isfinite(cell_values).all() or np.any(cell_values <= 0.0):
        raise ValueError("the scalar diffusion coefficient must be finite and strictly positive")
    return np.asarray(cell_values, dtype=float)


def _local_unit_stiffness(mesh: TriangleMesh) -> tuple[FloatArray, FloatArray]:
    points = mesh.points[mesh.triangles]
    edge_1 = points[:, 1] - points[:, 0]
    edge_2 = points[:, 2] - points[:, 0]
    g00 = np.einsum("ij,ij->i", edge_1, edge_1)
    g01 = np.einsum("ij,ij->i", edge_1, edge_2)
    g11 = np.einsum("ij,ij->i", edge_2, edge_2)
    determinant = g00 * g11 - g01 * g01
    if np.any(determinant <= 0.0):
        raise ValueError("a triangle has a singular intrinsic metric")
    area = 0.5 * np.sqrt(determinant)
    inverse_metric = np.empty((mesh.n_triangles, 2, 2), dtype=float)
    inverse_metric[:, 0, 0] = g11 / determinant
    inverse_metric[:, 0, 1] = -g01 / determinant
    inverse_metric[:, 1, 0] = -g01 / determinant
    inverse_metric[:, 1, 1] = g00 / determinant
    reference_gradients = np.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
    local = area[:, None, None] * np.einsum(
        "ia,eab,jb->eij", reference_gradients, inverse_metric, reference_gradients
    )
    return area, np.asarray(local, dtype=float)


def _assemble_local(mesh: TriangleMesh, local: FloatArray) -> csr_matrix:
    rows = np.repeat(mesh.triangles, 3, axis=1).ravel()
    columns = np.tile(mesh.triangles, (1, 3)).ravel()
    matrix = coo_matrix(
        (local.ravel(), (rows, columns)),
        shape=(mesh.n_vertices, mesh.n_vertices),
        dtype=float,
    ).tocsr()
    return _symmetrise(matrix)


def assemble_p1(
    mesh: TriangleMesh,
    coefficient: float | ArrayLike = 1.0,
    *,
    coefficient_location: str = "auto",
) -> SurfaceP1Operators:
    """Assemble consistent/lumped mass and scalar surface stiffness matrices.

    A nodal P1 coefficient is integrated exactly because the basis gradients
    are constant on each triangle, so only its cell average is needed.
    Strict positivity keeps each anchored Dirichlet block coercive.
    """

    quality = mesh.assert_valid()
    cell_coefficient = _coefficient_per_cell(mesh, coefficient, coefficient_location)
    area, local_unit = _local_unit_stiffness(mesh)
    unit_stiffness = _assemble_local(mesh, local_unit)
    stiffness = _assemble_local(mesh, local_unit * cell_coefficient[:, None, None])

    local_mass = np.broadcast_to(
        np.asarray(((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0)))[None, :, :],
        (mesh.n_triangles, 3, 3),
    ).copy()
    local_mass *= area[:, None, None] / 12.0
    mass = _assemble_local(mesh, local_mass)
    vertex_area = np.asarray(mass.sum(axis=1)).ravel()
    if not np.isfinite(vertex_area).all() or np.any(vertex_area <= 0.0):
        raise FloatingPointError("assembled lumped vertex areas are not strictly positive")
    lumped_mass = diags(vertex_area, format="csr")

    return SurfaceP1Operators(
        mesh=mesh,
        mass=mass,
        lumped_mass=lumped_mass,
        stiffness=stiffness,
        unit_stiffness=unit_stiffness,
        vertex_area=vertex_area,
        cell_area=area,
        cell_coefficient=cell_coefficient,
        quality=quality,
    )


def _node_set(nodes: ArrayLike, size: int, name: str) -> IntArray:
    raw = np.asarray(nodes)
    if raw.dtype == bool:
        if raw.shape != (size,):
            raise ValueError(f"Boolean {name} mask must have shape ({size},)")
        indices = np.flatnonzero(raw)
    else:
        if raw.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional index array or Boolean mask")
        if not np.issubdtype(raw.dtype, np.integer):
            rounded = np.rint(raw)
            if not np.array_equal(raw, rounded):
                raise ValueError(f"{name} contains a non-integer index")
            raw = rounded
        indices = np.asarray(raw, dtype=np.int64)
    indices = np.unique(indices)
    if indices.size == 0:
        raise ValueError(f"{name} is empty")
    if int(indices[0]) < 0 or int(indices[-1]) >= size:
        raise ValueError(f"{name} contains an out-of-range vertex index")
    return np.asarray(indices, dtype=np.int64)


def _safe_sparse_solve(matrix: csr_matrix, rhs: FloatArray, label: str) -> FloatArray:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        try:
            result = np.asarray(spsolve(matrix, rhs), dtype=float)
        except MatrixRankWarning as exc:
            raise np.linalg.LinAlgError(f"{label} is singular") from exc
    if not np.isfinite(result).all():
        raise np.linalg.LinAlgError(f"{label} solve returned a non-finite value")
    return result


@dataclass(frozen=True)
class _CapacityAlgebra:
    value: float
    source_flux: float
    sink_flux: float
    energy: float
    relative_residual: float
    flux_energy_defect: float
    potential: FloatArray


def _capacity_for_matrix(
    stiffness: csr_matrix,
    source: IntArray,
    sink: IntArray,
) -> _CapacityAlgebra:
    size = stiffness.shape[0]
    boundary = np.union1d(source, sink)
    free_mask = np.ones(size, dtype=bool)
    free_mask[boundary] = False
    free = np.flatnonzero(free_mask)
    potential = np.zeros(size, dtype=float)
    potential[source] = 1.0
    if free.size:
        free_matrix = stiffness[free][:, free].tocsr()
        rhs = -(stiffness[free][:, boundary] @ potential[boundary])
        interior = _safe_sparse_solve(free_matrix, np.asarray(rhs), "Dirichlet block")
        potential[free] = interior
        residual = free_matrix @ interior - rhs
        denominator = max(
            float(np.linalg.norm(rhs)),
            float(np.linalg.norm(free_matrix @ interior)),
            np.finfo(float).tiny,
        )
        relative_residual = float(np.linalg.norm(residual) / denominator)
    else:
        relative_residual = 0.0

    reaction = np.asarray(stiffness @ potential, dtype=float)
    source_flux = float(np.sum(reaction[source]))
    sink_flux = float(-np.sum(reaction[sink]))
    energy = float(potential @ reaction)
    value = 0.5 * (source_flux + sink_flux)
    scale = max(abs(value), abs(energy), np.finfo(float).tiny)
    defect = max(
        abs(source_flux - sink_flux),
        abs(source_flux - energy),
        abs(sink_flux - energy),
    ) / scale
    return _CapacityAlgebra(
        value=value,
        source_flux=source_flux,
        sink_flux=sink_flux,
        energy=energy,
        relative_residual=relative_residual,
        flux_energy_defect=float(defect),
        potential=potential,
    )


def solve_dirichlet_capacity(
    operators: SurfaceP1Operators,
    source_nodes: ArrayLike,
    sink_nodes: ArrayLike,
) -> SurfaceCapacityResult:
    """Solve a unit-potential capacity problem between two vertex sets.

    The source is held at one, the sink at zero, and all other exterior
    boundaries carry natural zero flux.  ``normalized_value`` is the capacity
    divided by the unit-coefficient capacity on the identical mesh and node
    sets.  It therefore removes mesh scale and electrode geometry while
    retaining the effect of the supplied scalar coefficient.
    """

    source = _node_set(source_nodes, operators.n_vertices, "source_nodes")
    sink = _node_set(sink_nodes, operators.n_vertices, "sink_nodes")
    overlap = np.intersect1d(source, sink)
    if overlap.size:
        raise ValueError(f"source and sink overlap at {overlap.size} vertices")

    component_count, component_labels = operators.mesh.component_labels()
    boundary_components = set(component_labels[np.union1d(source, sink)].tolist())
    all_components = set(range(component_count))
    unanchored = all_components.difference(boundary_components)
    if unanchored:
        raise ValueError(
            f"{len(unanchored)} mesh component(s) have no Dirichlet vertex; "
            "the free stiffness block would be singular"
        )
    source_components = set(component_labels[source].tolist())
    sink_components = set(component_labels[sink].tolist())
    if source_components.isdisjoint(sink_components):
        raise ValueError("no connected mesh component contains both source and sink")

    result = _capacity_for_matrix(operators.stiffness, source, sink)
    reference = _capacity_for_matrix(operators.unit_stiffness, source, sink)
    if not np.isfinite(reference.value) or reference.value <= 0.0:
        raise FloatingPointError("unit-coefficient reference capacity is not positive")
    if not np.isfinite(result.value) or result.value <= 0.0:
        raise FloatingPointError("surface capacity is not positive")
    return SurfaceCapacityResult(
        value=result.value,
        normalized_value=float(result.value / reference.value),
        reference_capacity=reference.value,
        source_flux=result.source_flux,
        sink_flux=result.sink_flux,
        energy=result.energy,
        relative_residual=result.relative_residual,
        flux_energy_defect=result.flux_energy_defect,
        potential=result.potential,
        source_nodes=source,
        sink_nodes=sink,
    )


def solve_labelled_capacity(
    operators: SurfaceP1Operators,
    field_name: str,
    source_labels: int | float | Sequence[int | float],
    sink_labels: int | float | Sequence[int | float],
    *,
    atol: float = 0.0,
) -> SurfaceCapacityResult:
    """Capacity solve using two values (or value groups) in point data."""

    source = operators.mesh.labelled_nodes(field_name, source_labels, atol=atol)
    sink = operators.mesh.labelled_nodes(field_name, sink_labels, atol=atol)
    return solve_dirichlet_capacity(operators, source, sink)


__all__ = [
    "SurfaceCapacityResult",
    "SurfaceP1Operators",
    "assemble_p1",
    "solve_dirichlet_capacity",
    "solve_labelled_capacity",
]
