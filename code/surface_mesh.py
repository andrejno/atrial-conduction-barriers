"""Triangular-surface meshes and geometry checks.

The surface finite-element backend intentionally has a small dependency
footprint.  Synthetic meshes need only NumPy and SciPy.  Reading VTK or VTU
files is delegated to the optional :mod:`meshio` package so that the parser is
not tied to one legacy-VTK dialect.

Only linear, three-node triangles enter the surface representation.  A
linear-tetrahedron VTK grid can also be ingested by extracting its exterior
faces.  The checks in
``TriangleMesh.assert_valid`` reject geometry for which a scalar P1 surface
problem would be ambiguous (degenerate or duplicate cells, isolated points,
and non-manifold edges).  A consistently oriented mesh is reported but is not
required by the orientation-independent scalar assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _import_meshio():
    try:
        import meshio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "VTK mesh ingestion requires the optional package 'meshio>=5.3' "
            "(for example: python -m pip install 'meshio>=5.3')."
        ) from exc
    return meshio


def inspect_mesh_file(path: str | Path) -> dict[str, object]:
    """Return a value-free inventory of a mesh file without constructing it.

    The inventory is suitable for an inspect-only first pass: it contains cell
    types/counts and data-array names, shapes, and dtypes, but no field values.
    """

    meshio = _import_meshio()
    mesh_path = Path(path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh file not found: {mesh_path}")
    try:
        raw = meshio.read(mesh_path)
    except Exception as exc:
        raise ValueError(f"meshio could not read {mesh_path}: {exc}") from exc
    cell_blocks: list[dict[str, object]] = []
    for block_index, block in enumerate(raw.cells):
        data = np.asarray(block.data)
        cell_blocks.append(
            {
                "block": block_index,
                "type": str(block.type),
                "count": int(data.shape[0]),
                "nodes_per_cell": int(data.shape[1]) if data.ndim == 2 else None,
            }
        )
    point_fields = {
        str(name): {"shape": list(np.asarray(values).shape), "dtype": str(np.asarray(values).dtype)}
        for name, values in raw.point_data.items()
    }
    cell_fields: dict[str, list[dict[str, object]]] = {}
    for name, arrays in raw.cell_data.items():
        cell_fields[str(name)] = [
            {
                "block": block_index,
                "shape": list(np.asarray(values).shape),
                "dtype": str(np.asarray(values).dtype),
            }
            for block_index, values in enumerate(arrays)
        ]
    return {
        "path": str(mesh_path),
        "n_points": int(np.asarray(raw.points).shape[0]),
        "point_dimension": int(np.asarray(raw.points).shape[1]),
        "cell_blocks": cell_blocks,
        "point_data": point_fields,
        "cell_data": cell_fields,
    }


def _tetrahedral_boundary(
    points: FloatArray,
    tetrahedra: IntArray,
) -> tuple[IntArray, IntArray, IntArray]:
    """Extract outward-oriented exterior faces and compact their vertex IDs."""

    xyz = np.asarray(points, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3) or not np.isfinite(xyz).all():
        raise ValueError("tetrahedral points must be finite 3-D coordinates")
    if xyz.shape[1] == 2:
        xyz = np.column_stack((xyz, np.zeros(xyz.shape[0])))
    if cells.ndim != 2 or cells.shape[1] != 4 or cells.shape[0] == 0:
        raise ValueError("tetra connectivity must have shape (n_tetrahedra, 4)")
    if int(np.min(cells)) < 0 or int(np.max(cells)) >= xyz.shape[0]:
        raise ValueError("tetra connectivity contains an out-of-range vertex index")
    repeated = np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0, axis=1)
    if np.any(repeated):
        raise ValueError(f"tetrahedral grid has {int(np.sum(repeated))} repeated-index cells")

    tetra_points = xyz[cells]
    volumes6 = np.abs(
        np.einsum(
            "ij,ij->i",
            tetra_points[:, 1] - tetra_points[:, 0],
            np.cross(
                tetra_points[:, 2] - tetra_points[:, 0],
                tetra_points[:, 3] - tetra_points[:, 0],
            ),
        )
    )
    extent = np.ptp(xyz, axis=0)
    length_scale = max(float(np.linalg.norm(extent)), np.finfo(float).tiny)
    volume6_tolerance = 768.0 * np.finfo(float).eps * length_scale**3
    degenerate = volumes6 <= volume6_tolerance
    if np.any(degenerate):
        raise ValueError(
            f"tetrahedral grid has {int(np.sum(degenerate))} degenerate cells "
            f"at tolerance {volume6_tolerance / 6.0:.3e}"
        )

    local_faces = np.asarray(
        ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)), dtype=np.int64
    )
    opposite_local = np.asarray((0, 1, 2, 3), dtype=np.int64)
    faces = cells[:, local_faces].reshape(-1, 3)
    opposite = cells[:, opposite_local].reshape(-1)
    parents = np.repeat(np.arange(cells.shape[0], dtype=np.int64), 4)
    canonical = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(
        canonical, axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError(
            f"tetrahedral grid has {int(np.sum(counts > 2))} non-manifold faces "
            "incident to more than two cells"
        )
    exterior_mask = counts[inverse] == 1
    exterior = faces[exterior_mask].copy()
    exterior_opposite = opposite[exterior_mask]
    exterior_parents = parents[exterior_mask]
    if exterior.shape[0] == 0:
        raise ValueError("tetrahedral grid has no exterior faces")

    face_points = xyz[exterior]
    normals = np.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    )
    toward_interior = xyz[exterior_opposite] - face_points[:, 0]
    inward = np.einsum("ij,ij->i", normals, toward_interior) > 0.0
    exterior[inward, 1], exterior[inward, 2] = (
        exterior[inward, 2].copy(),
        exterior[inward, 1].copy(),
    )

    original_vertex_ids = np.unique(exterior.ravel()).astype(np.int64)
    remap = np.full(xyz.shape[0], -1, dtype=np.int64)
    remap[original_vertex_ids] = np.arange(original_vertex_ids.size)
    return remap[exterior], exterior_parents, original_vertex_ids


@dataclass(frozen=True)
class MeshQualityReport:
    """Geometry and topology diagnostics for a triangular surface."""

    n_vertices: int
    n_triangles: int
    n_used_vertices: int
    n_edges: int
    connected_components: int
    boundary_edges: int
    boundary_components: int
    boundary_branch_vertices: int
    nonmanifold_edges: int
    nonmanifold_vertices: int
    orientation_conflicts: int
    isolated_vertices: int
    repeated_index_triangles: int
    duplicate_triangles: int
    degenerate_triangles: int
    euler_characteristic: int
    total_area: float
    minimum_area: float
    maximum_area: float
    minimum_angle_degrees: float
    maximum_edge_ratio: float
    minimum_mean_ratio_quality: float
    area_tolerance: float

    @property
    def valid_for_p1(self) -> bool:
        """Whether the mesh is admissible for the scalar P1 assembly."""

        return not any(
            (
                self.isolated_vertices,
                self.repeated_index_triangles,
                self.duplicate_triangles,
                self.degenerate_triangles,
                self.nonmanifold_edges,
                self.nonmanifold_vertices,
            )
        )

    def problems(self, require_connected: bool = False) -> list[str]:
        """Return human-readable failures, leaving quality warnings numeric."""

        failures: list[str] = []
        for count, name in (
            (self.repeated_index_triangles, "triangles with repeated vertex indices"),
            (self.duplicate_triangles, "duplicate triangles"),
            (self.degenerate_triangles, "degenerate triangles"),
            (self.isolated_vertices, "isolated vertices"),
            (self.nonmanifold_edges, "non-manifold edges"),
            (self.nonmanifold_vertices, "non-manifold vertex links"),
        ):
            if count:
                failures.append(f"{count} {name}")
        if require_connected and self.connected_components != 1:
            failures.append(
                f"{self.connected_components} connected components (one required)"
            )
        return failures


@dataclass
class TriangleMesh:
    """A linear triangular surface embedded in three-dimensional space."""

    points: ArrayLike
    triangles: ArrayLike
    point_data: Mapping[str, ArrayLike] = field(default_factory=dict)
    cell_data: Mapping[str, ArrayLike] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("points must have shape (n_vertices, 2) or (n_vertices, 3)")
        if points.shape[0] < 3:
            raise ValueError("a triangular mesh needs at least three vertices")
        if points.shape[1] == 2:
            points = np.column_stack((points, np.zeros(points.shape[0], dtype=float)))
        if not np.isfinite(points).all():
            raise ValueError("mesh points contain a non-finite coordinate")

        raw_triangles = np.asarray(self.triangles)
        if raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (n_triangles, 3)")
        if raw_triangles.shape[0] == 0:
            raise ValueError("the mesh has no triangles")
        if not np.issubdtype(raw_triangles.dtype, np.integer):
            rounded = np.rint(raw_triangles)
            if not np.array_equal(raw_triangles, rounded):
                raise ValueError("triangle connectivity must contain integer indices")
            raw_triangles = rounded
        triangles = np.asarray(raw_triangles, dtype=np.int64)
        if int(np.min(triangles)) < 0 or int(np.max(triangles)) >= points.shape[0]:
            raise ValueError("triangle connectivity contains an out-of-range vertex index")

        point_data: dict[str, NDArray[np.generic]] = {}
        for name, values in self.point_data.items():
            array = np.asarray(values)
            if array.ndim == 0 or array.shape[0] != points.shape[0]:
                raise ValueError(
                    f"point_data[{name!r}] must have first dimension {points.shape[0]}"
                )
            point_data[str(name)] = array.copy()

        cell_data: dict[str, NDArray[np.generic]] = {}
        for name, values in self.cell_data.items():
            array = np.asarray(values)
            if array.ndim == 0 or array.shape[0] != triangles.shape[0]:
                raise ValueError(
                    f"cell_data[{name!r}] must have first dimension {triangles.shape[0]}"
                )
            cell_data[str(name)] = array.copy()

        self.points = np.ascontiguousarray(points, dtype=float)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.int64)
        self.point_data = point_data
        self.cell_data = cell_data

    @property
    def n_vertices(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        cell_source: str = "auto",
    ) -> "TriangleMesh":
        """Read a VTK/VTU triangular surface or tetrahedral volume mesh.

        ``meshio`` is optional because the numerical verification suite creates
        its meshes directly.  Install ``meshio>=5.3`` when ingesting legacy VTK
        unstructured grids.  Vertex data arrays are retained.  With
        ``cell_source='tetra_boundary'``, exterior faces of linear tetrahedra
        are extracted and tetrahedral cell fields are inherited by their
        boundary faces.  ``'auto'`` accepts exactly one of a triangle surface
        or a tetrahedral volume; a file containing both is deliberately
        rejected as ambiguous.  Use ``'triangles'`` or ``'tetra_boundary'`` to
        resolve such a file explicitly.  Higher-order and unsupported 2-D/3-D
        cells are rejected rather than silently dropped or linearised.
        """

        meshio = _import_meshio()

        mesh_path = Path(path)
        if not mesh_path.is_file():
            raise FileNotFoundError(f"mesh file not found: {mesh_path}")
        try:
            raw = meshio.read(mesh_path)
        except Exception as exc:
            raise ValueError(f"meshio could not read {mesh_path}: {exc}") from exc

        if cell_source not in {"auto", "triangles", "tetra_boundary"}:
            raise ValueError(
                "cell_source must be 'auto', 'triangles', or 'tetra_boundary'"
            )

        triangle_block_indices: list[int] = []
        tetra_block_indices: list[int] = []
        triangle_blocks: list[IntArray] = []
        tetra_blocks: list[IntArray] = []
        unsupported_types: list[str] = []
        harmless_auxiliary = {"vertex", "line"}
        for block_index, block in enumerate(raw.cells):
            if block.type == "triangle":
                triangle_block_indices.append(block_index)
                triangle_blocks.append(np.asarray(block.data, dtype=np.int64))
            elif block.type == "tetra":
                tetra_block_indices.append(block_index)
                tetra_blocks.append(np.asarray(block.data, dtype=np.int64))
            elif block.type not in harmless_auxiliary and np.asarray(block.data).size:
                unsupported_types.append(block.type)
        if unsupported_types:
            kinds = ", ".join(sorted(set(unsupported_types)))
            raise ValueError(
                f"unsupported non-linear or non-simplicial cell blocks ({kinds}); "
                "provide linear triangle or tetra cells"
            )
        if cell_source == "auto":
            if triangle_blocks and tetra_blocks:
                raise ValueError(
                    "the file contains both triangle and tetra cells; choose "
                    "cell_source='triangles' or cell_source='tetra_boundary' explicitly"
                )
            selected_source = "triangles" if triangle_blocks else "tetra_boundary"
        else:
            selected_source = cell_source

        if selected_source == "triangles" and not triangle_blocks:
            available = ", ".join(sorted({block.type for block in raw.cells})) or "none"
            raise ValueError(
                f"{mesh_path} contains no linear triangle cells; cell types: {available}"
            )
        if selected_source == "tetra_boundary" and not tetra_blocks:
            available = ", ".join(sorted({block.type for block in raw.cells})) or "none"
            raise ValueError(
                f"{mesh_path} contains no linear tetra cells; cell types: {available}"
            )

        triangle_data: dict[str, NDArray[np.generic]] = {}
        output_points = np.asarray(raw.points, dtype=float)
        output_point_data = {
            str(k): np.asarray(v) for k, v in raw.point_data.items()
        }
        if selected_source == "triangles":
            triangles = np.vstack(triangle_blocks)
            for name, arrays in raw.cell_data.items():
                selected = [np.asarray(arrays[i]) for i in triangle_block_indices]
                if selected:
                    triangle_data[str(name)] = np.concatenate(selected, axis=0)
        else:
            tetrahedra = np.vstack(tetra_blocks)
            tetra_data: dict[str, NDArray[np.generic]] = {}
            for name, arrays in raw.cell_data.items():
                selected = [np.asarray(arrays[i]) for i in tetra_block_indices]
                if selected:
                    tetra_data[str(name)] = np.concatenate(selected, axis=0)
            triangles, parent_tetrahedra, original_vertex_ids = _tetrahedral_boundary(
                np.asarray(raw.points, dtype=float), tetrahedra
            )
            output_points = output_points[original_vertex_ids]
            output_point_data = {
                name: values[original_vertex_ids]
                for name, values in output_point_data.items()
            }
            output_point_data["original_vertex_id"] = original_vertex_ids
            triangle_data["parent_tetrahedron"] = parent_tetrahedra
            for name, values in tetra_data.items():
                triangle_data[name] = np.asarray(values)[parent_tetrahedra]
        return cls(
            points=output_points,
            triangles=triangles,
            point_data=output_point_data,
            cell_data=triangle_data,
        )

    def triangle_areas(self) -> FloatArray:
        p = self.points[self.triangles]
        return 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
        )

    def _edge_table(self) -> tuple[IntArray, IntArray, IntArray]:
        directed = np.vstack(
            (
                self.triangles[:, [0, 1]],
                self.triangles[:, [1, 2]],
                self.triangles[:, [2, 0]],
            )
        )
        undirected = np.sort(directed, axis=1)
        unique, inverse, counts = np.unique(
            undirected, axis=0, return_inverse=True, return_counts=True
        )
        signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
        signed_counts = np.bincount(inverse, weights=signs, minlength=unique.shape[0])
        return unique, counts.astype(np.int64), np.asarray(signed_counts, dtype=np.int64)

    def component_labels(self) -> tuple[int, IntArray]:
        """Return vertex-component count and zero-based component labels."""

        edges, _, _ = self._edge_table()
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        cols = np.concatenate((edges[:, 1], edges[:, 0]))
        graph = coo_matrix(
            (np.ones(rows.size, dtype=np.int8), (rows, cols)),
            shape=(self.n_vertices, self.n_vertices),
        ).tocsr()
        count, labels = connected_components(graph, directed=False, return_labels=True)
        return int(count), np.asarray(labels, dtype=np.int64)

    def boundary_edges(self) -> IntArray:
        edges, counts, _ = self._edge_table()
        return np.asarray(edges[counts == 1], dtype=np.int64)

    def boundary_nodes(self) -> IntArray:
        edges = self.boundary_edges()
        if edges.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(edges.ravel()).astype(np.int64)

    def labelled_nodes(
        self,
        field_name: str,
        labels: int | float | Sequence[int | float],
        *,
        atol: float = 0.0,
    ) -> IntArray:
        """Select vertices whose scalar point-data field has a requested label."""

        if field_name not in self.point_data:
            available = ", ".join(sorted(self.point_data)) or "none"
            raise KeyError(f"point-data field {field_name!r} not found; available: {available}")
        values = np.asarray(self.point_data[field_name])
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError(f"point-data field {field_name!r} is not scalar")
        requested = np.atleast_1d(np.asarray(labels))
        mask = np.zeros(self.n_vertices, dtype=bool)
        for label in requested:
            if atol > 0.0 and np.issubdtype(values.dtype, np.number):
                mask |= np.isclose(values.astype(float), float(label), rtol=0.0, atol=atol)
            else:
                mask |= values == label
        return np.flatnonzero(mask).astype(np.int64)

    def quality_report(self, area_tolerance: float | None = None) -> MeshQualityReport:
        p = self.points[self.triangles]
        edge_vectors = np.stack(
            (p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]),
            axis=1,
        )
        edge_lengths = np.linalg.norm(edge_vectors, axis=2)
        areas = self.triangle_areas()
        length_scale = max(float(np.max(edge_lengths)), np.finfo(float).tiny)
        if area_tolerance is None:
            area_tolerance = 128.0 * np.finfo(float).eps * length_scale**2
        if not np.isfinite(area_tolerance) or area_tolerance < 0.0:
            raise ValueError("area_tolerance must be finite and non-negative")

        repeated = np.sum(
            (self.triangles[:, 0] == self.triangles[:, 1])
            | (self.triangles[:, 1] == self.triangles[:, 2])
            | (self.triangles[:, 2] == self.triangles[:, 0])
        )
        sorted_cells = np.sort(self.triangles, axis=1)
        duplicate_count = int(
            self.n_triangles - np.unique(sorted_cells, axis=0).shape[0]
        )

        edges, edge_counts, edge_signs = self._edge_table()
        used = np.unique(self.triangles)
        isolated = self.n_vertices - int(used.size)
        component_count, _ = self.component_labels()
        boundary = edges[edge_counts == 1]
        nonmanifold = int(np.sum(edge_counts > 2))
        orientation_conflicts = int(np.sum((edge_counts == 2) & (edge_signs != 0)))

        boundary_components = 0
        boundary_branch_vertices = 0
        if boundary.size:
            boundary_degree = np.bincount(boundary.ravel(), minlength=self.n_vertices)
            boundary_branch_vertices = int(np.sum((boundary_degree != 0) & (boundary_degree != 2)))
            bnodes = np.flatnonzero(boundary_degree)
            remap = np.full(self.n_vertices, -1, dtype=np.int64)
            remap[bnodes] = np.arange(bnodes.size)
            brows = np.concatenate((remap[boundary[:, 0]], remap[boundary[:, 1]]))
            bcols = np.concatenate((remap[boundary[:, 1]], remap[boundary[:, 0]]))
            bgraph = coo_matrix(
                (np.ones(brows.size, dtype=np.int8), (brows, bcols)),
                shape=(bnodes.size, bnodes.size),
            ).tocsr()
            boundary_components = int(
                connected_components(bgraph, directed=False, return_labels=False)
            )

        # With manifold edges, a vertex link must be one cycle (interior) or
        # one path (boundary).  This catches bow-tie contacts and disconnected
        # triangle fans that an edge-incidence test alone cannot see.
        incident_faces: list[list[int]] = [[] for _ in range(self.n_vertices)]
        for face_index, triangle in enumerate(self.triangles):
            for vertex in triangle:
                incident_faces[int(vertex)].append(face_index)
        nonmanifold_vertices = 0
        for vertex, face_indices in enumerate(incident_faces):
            if not face_indices:
                continue
            link_adjacency: dict[int, set[int]] = {}
            for face_index in face_indices:
                triangle = self.triangles[face_index]
                others = triangle[triangle != vertex]
                if others.size != 2:
                    continue
                left, right = int(others[0]), int(others[1])
                link_adjacency.setdefault(left, set()).add(right)
                link_adjacency.setdefault(right, set()).add(left)
            degrees = [len(neighbours) for neighbours in link_adjacency.values()]
            if not degrees:
                nonmanifold_vertices += 1
                continue
            remaining = set(link_adjacency)
            link_components = 0
            while remaining:
                link_components += 1
                stack = [remaining.pop()]
                while stack:
                    current = stack.pop()
                    unseen = link_adjacency[current].intersection(remaining)
                    remaining.difference_update(unseen)
                    stack.extend(unseen)
            degree_one = sum(degree == 1 for degree in degrees)
            degree_two = sum(degree == 2 for degree in degrees)
            valid_link = link_components == 1 and (
                degree_two == len(degrees)
                or (degree_one == 2 and degree_one + degree_two == len(degrees))
            )
            if not valid_link:
                nonmanifold_vertices += 1

        valid_geometry = (areas > area_tolerance) & np.all(edge_lengths > 0.0, axis=1)
        minimum_angle = 0.0
        maximum_edge_ratio = float("inf")
        minimum_quality = 0.0
        if np.any(valid_geometry):
            lengths = edge_lengths[valid_geometry]
            triangle_areas = areas[valid_geometry]
            a = lengths[:, 1]
            b = lengths[:, 2]
            c = lengths[:, 0]
            denominators = np.stack((2.0 * b * c, 2.0 * c * a, 2.0 * a * b), axis=1)
            numerators = np.stack(
                (b * b + c * c - a * a, c * c + a * a - b * b, a * a + b * b - c * c),
                axis=1,
            )
            cosines = np.clip(numerators / denominators, -1.0, 1.0)
            minimum_angle = float(np.degrees(np.min(np.arccos(cosines))))
            maximum_edge_ratio = float(np.max(np.max(lengths, axis=1) / np.min(lengths, axis=1)))
            qualities = 4.0 * np.sqrt(3.0) * triangle_areas / np.sum(lengths**2, axis=1)
            minimum_quality = float(np.min(qualities))

        return MeshQualityReport(
            n_vertices=self.n_vertices,
            n_triangles=self.n_triangles,
            n_used_vertices=int(used.size),
            n_edges=int(edges.shape[0]),
            connected_components=component_count,
            boundary_edges=int(boundary.shape[0]),
            boundary_components=boundary_components,
            boundary_branch_vertices=boundary_branch_vertices,
            nonmanifold_edges=nonmanifold,
            nonmanifold_vertices=nonmanifold_vertices,
            orientation_conflicts=orientation_conflicts,
            isolated_vertices=isolated,
            repeated_index_triangles=int(repeated),
            duplicate_triangles=duplicate_count,
            degenerate_triangles=int(np.sum(~valid_geometry)),
            euler_characteristic=int(used.size - edges.shape[0] + self.n_triangles),
            total_area=float(np.sum(areas)),
            minimum_area=float(np.min(areas)),
            maximum_area=float(np.max(areas)),
            minimum_angle_degrees=minimum_angle,
            maximum_edge_ratio=maximum_edge_ratio,
            minimum_mean_ratio_quality=minimum_quality,
            area_tolerance=float(area_tolerance),
        )

    def assert_valid(
        self,
        *,
        require_connected: bool = False,
        require_consistent_orientation: bool = False,
        area_tolerance: float | None = None,
    ) -> MeshQualityReport:
        """Validate P1 prerequisites and return the complete quality report."""

        report = self.quality_report(area_tolerance=area_tolerance)
        failures = report.problems(require_connected=require_connected)
        if require_consistent_orientation and report.orientation_conflicts:
            failures.append(f"{report.orientation_conflicts} shared-edge orientation conflicts")
        if failures:
            raise ValueError("invalid triangular surface: " + "; ".join(failures))
        return report


def rectangular_tri_mesh(
    nx: int,
    ny: int,
    *,
    length: float = 2.0,
    width: float = 1.0,
) -> TriangleMesh:
    """Structured planar rectangle with labelled left/right electrodes."""

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive cell counts")
    if not (np.isfinite(length) and np.isfinite(width) and length > 0.0 and width > 0.0):
        raise ValueError("length and width must be positive and finite")
    x = np.linspace(0.0, length, nx + 1)
    y = np.linspace(0.0, width, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size)))

    def vertex(i: int, j: int) -> int:
        return i * (ny + 1) + j

    triangles: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            a, b = vertex(i, j), vertex(i + 1, j)
            c, d = vertex(i + 1, j + 1), vertex(i, j + 1)
            if (i + j) % 2 == 0:
                triangles.extend(((a, b, c), (a, c, d)))
            else:
                triangles.extend(((a, b, d), (b, c, d)))
    boundary_id = np.zeros(points.shape[0], dtype=np.int16)
    boundary_id[np.isclose(points[:, 0], 0.0)] = 1
    boundary_id[np.isclose(points[:, 0], length)] = 2
    return TriangleMesh(points, np.asarray(triangles), {"boundary_id": boundary_id})


def annular_tri_mesh(
    n_radial: int,
    n_angular: int,
    *,
    inner_radius: float = 1.0,
    outer_radius: float = 2.0,
) -> TriangleMesh:
    """Conforming planar annulus with labelled inner/outer electrodes."""

    if n_radial < 1 or n_angular < 3:
        raise ValueError("require n_radial >= 1 and n_angular >= 3")
    if not (
        np.isfinite(inner_radius)
        and np.isfinite(outer_radius)
        and 0.0 < inner_radius < outer_radius
    ):
        raise ValueError("require finite radii satisfying 0 < inner_radius < outer_radius")
    radii = np.geomspace(inner_radius, outer_radius, n_radial + 1)
    angles = 2.0 * np.pi * np.arange(n_angular) / n_angular
    rr, tt = np.meshgrid(radii, angles, indexing="ij")
    points = np.column_stack(
        ((rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel(), np.zeros(rr.size))
    )

    def vertex(i: int, j: int) -> int:
        return i * n_angular + (j % n_angular)

    triangles: list[tuple[int, int, int]] = []
    for i in range(n_radial):
        for j in range(n_angular):
            a, b = vertex(i, j), vertex(i + 1, j)
            c, d = vertex(i + 1, j + 1), vertex(i, j + 1)
            triangles.extend(((a, b, c), (a, c, d)))
    boundary_id = np.zeros(points.shape[0], dtype=np.int16)
    boundary_id[:n_angular] = 1
    boundary_id[-n_angular:] = 2
    return TriangleMesh(points, np.asarray(triangles), {"boundary_id": boundary_id})


def icosphere_tri_mesh(refinement_level: int, *, radius: float = 1.0) -> TriangleMesh:
    """Closed, outward-oriented icosphere for curved-surface verification."""

    if not isinstance(refinement_level, int) or refinement_level < 0:
        raise ValueError("refinement_level must be a non-negative integer")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    points = np.asarray(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=float,
    )
    points *= radius / np.linalg.norm(points, axis=1)[:, None]
    triangles = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )
    for _ in range(refinement_level):
        midpoint_cache: dict[tuple[int, int], int] = {}
        refined_points = points.tolist()
        refined_triangles: list[tuple[int, int, int]] = []

        def midpoint(left: int, right: int) -> int:
            key = tuple(sorted((int(left), int(right))))
            if key not in midpoint_cache:
                value = 0.5 * (points[left] + points[right])
                value *= radius / np.linalg.norm(value)
                midpoint_cache[key] = len(refined_points)
                refined_points.append(value.tolist())
            return midpoint_cache[key]

        for a, b, c in triangles:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined_triangles.extend(
                ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))
            )
        points = np.asarray(refined_points, dtype=float)
        triangles = np.asarray(refined_triangles, dtype=np.int64)
    mesh = TriangleMesh(points, triangles)
    mesh.assert_valid(require_connected=True, require_consistent_orientation=True)
    return mesh


__all__ = [
    "MeshQualityReport",
    "TriangleMesh",
    "annular_tri_mesh",
    "icosphere_tri_mesh",
    "inspect_mesh_file",
    "rectangular_tri_mesh",
]
