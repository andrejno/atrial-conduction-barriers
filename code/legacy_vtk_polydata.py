"""Small, strict reader for legacy VTK ``POLYDATA`` surface files.

The patient-derived ERP meshes used by the reproducibility package are legacy
VTK ``POLYDATA`` files.  They contain a mixture of triangles and quadrilaterals
and use both the classic polygon encoding (VTK 4.x) and the offset/connectivity
encoding introduced in VTK 5.1.  ``meshio`` deliberately does not support this
legacy ``POLYDATA`` dialect, so this module handles the narrow format directly.

Only surface polygons and numeric point/cell attributes are accepted.  Both
ASCII and big-endian binary payloads are supported.  Polygons are converted to
triangles; a quadrilateral is split along its shorter diagonal, with a
lexicographic vertex-ID tie break.  Polygon cell fields are copied to every
child triangle.  The added ``source_polygon_id`` and ``source_polygon_size``
arrays make that conversion explicit and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DataArray = NDArray[np.generic]
PolyDataResult = tuple[
    FloatArray,
    IntArray,
    dict[str, DataArray],
    dict[str, DataArray],
]


@dataclass(frozen=True)
class SurfaceCleanupReport:
    """Exact record of deterministic triangle removal and vertex compaction."""

    input_points: int
    output_points: int
    input_triangles: int
    output_triangles: int
    area_tolerance: float
    removed_zero_area_triangle_indices: tuple[int, ...]
    removed_duplicate_triangle_indices: tuple[int, ...]
    removed_source_polygon_ids: tuple[int, ...]
    removed_isolated_point_indices: tuple[int, ...]

    @property
    def removed_triangles(self) -> int:
        return self.input_triangles - self.output_triangles

    @property
    def removed_points(self) -> int:
        return self.input_points - self.output_points


_VTK_DTYPES: dict[str, np.dtype] = {
    "char": np.dtype("i1"),
    "signed_char": np.dtype("i1"),
    "unsigned_char": np.dtype("u1"),
    "short": np.dtype("i2"),
    "unsigned_short": np.dtype("u2"),
    "int": np.dtype("i4"),
    "unsigned_int": np.dtype("u4"),
    "long": np.dtype("i8"),
    "unsigned_long": np.dtype("u8"),
    "long_long": np.dtype("i8"),
    "unsigned_long_long": np.dtype("u8"),
    "vtkidtype": np.dtype("i8"),
    "vtktypeint64": np.dtype("i8"),
    "vtktypeuint64": np.dtype("u8"),
    "float": np.dtype("f4"),
    "double": np.dtype("f8"),
}


class _LegacyReader:
    """Stateful reader whose numeric payload mode is fixed by the VTK header."""

    def __init__(self, handle: BinaryIO, *, binary: bool, path: Path) -> None:
        self.handle = handle
        self.binary = binary
        self.path = path

    def line(self, *, allow_eof: bool = False) -> str | None:
        raw = self.handle.readline()
        if raw == b"":
            if allow_eof:
                return None
            raise ValueError(f"unexpected end of file in {self.path}")
        try:
            return raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"expected an ASCII VTK directive at byte {self.handle.tell() - len(raw)} "
                f"in {self.path}"
            ) from exc

    def nonempty_line(self, *, allow_eof: bool = False) -> str | None:
        while True:
            value = self.line(allow_eof=allow_eof)
            if value is None or value:
                return value

    def _dtype(self, vtk_name: str, *, binary_byte_order: bool) -> np.dtype:
        key = vtk_name.lower()
        if key not in _VTK_DTYPES:
            supported = ", ".join(sorted(_VTK_DTYPES))
            raise ValueError(
                f"unsupported VTK numeric type {vtk_name!r} in {self.path}; "
                f"supported types: {supported}"
            )
        dtype = _VTK_DTYPES[key]
        if binary_byte_order and dtype.itemsize > 1:
            return dtype.newbyteorder(">")
        return dtype.newbyteorder("=")

    def values(self, count: int, vtk_name: str) -> DataArray:
        if count < 0:
            raise ValueError(f"negative numeric-array length in {self.path}")
        native_dtype = self._dtype(vtk_name, binary_byte_order=False)
        if count == 0:
            return np.empty(0, dtype=native_dtype)
        if self.binary:
            file_dtype = self._dtype(vtk_name, binary_byte_order=True)
            required = count * file_dtype.itemsize
            payload = self.handle.read(required)
            if len(payload) != required:
                raise ValueError(
                    f"truncated binary array in {self.path}: expected {required} bytes, "
                    f"read {len(payload)}"
                )
            result = np.frombuffer(payload, dtype=file_dtype, count=count).astype(
                native_dtype, copy=True
            )
            terminator = self.handle.read(1)
            if terminator == b"\r":
                terminator += self.handle.read(1)
            if terminator not in {b"\n", b"\r\n"}:
                raise ValueError(
                    f"binary array is not followed by a line break in {self.path}"
                )
            return result

        chunks: list[DataArray] = []
        received = 0
        while received < count:
            raw = self.line()
            assert raw is not None
            if not raw:
                continue
            chunk = np.fromstring(raw, sep=" ", dtype=native_dtype)
            if chunk.size == 0:
                raise ValueError(
                    f"expected {count - received} more numeric values, found {raw!r} "
                    f"in {self.path}"
                )
            if received + chunk.size > count:
                raise ValueError(
                    f"numeric line contains {received + chunk.size - count} excess values "
                    f"in {self.path}"
                )
            chunks.append(chunk)
            received += int(chunk.size)
        return np.concatenate(chunks).astype(native_dtype, copy=False)

    def next_is(self, prefix: bytes) -> bool:
        position = self.handle.tell()
        probe = self.handle.read(len(prefix))
        self.handle.seek(position)
        return probe == prefix


def _parse_header(line: str, keyword: str, path: Path) -> list[str]:
    tokens = line.split()
    if not tokens or tokens[0].upper() != keyword:
        raise ValueError(f"expected {keyword} directive, found {line!r} in {path}")
    return tokens


def _positive_count(token: str, label: str, path: Path) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise ValueError(f"invalid {label} {token!r} in {path}") from exc
    if value <= 0:
        raise ValueError(f"{label} must be positive in {path}, found {value}")
    return value


def _decode_polygons(
    reader: _LegacyReader,
    *,
    header_count: int,
    payload_size: int,
) -> tuple[IntArray, IntArray, int]:
    """Return offsets, connectivity, and polygon count for either encoding.

    In classic VTK files the first integer on ``POLYGONS`` is the number of
    polygons.  In the VTK 5.1 offset/connectivity layout it is instead the
    number of offsets, including both endpoints, so the polygon count is one
    smaller.  This distinction is easy to miss because both layouts retain the
    same ``POLYGONS`` keyword.
    """

    if reader.next_is(b"OFFSETS "):
        offset_header = _parse_header(reader.nonempty_line() or "", "OFFSETS", reader.path)
        if len(offset_header) != 2:
            raise ValueError(f"malformed OFFSETS directive in {reader.path}")
        if header_count < 2:
            raise ValueError(f"offset/connectivity POLYGONS needs at least two offsets")
        offsets = np.asarray(reader.values(header_count, offset_header[1]), dtype=np.int64)
        connection_header = _parse_header(
            reader.nonempty_line() or "", "CONNECTIVITY", reader.path
        )
        if len(connection_header) != 2:
            raise ValueError(f"malformed CONNECTIVITY directive in {reader.path}")
        connectivity = np.asarray(
            reader.values(payload_size, connection_header[1]), dtype=np.int64
        )
        if offsets[0] != 0 or offsets[-1] != payload_size:
            raise ValueError(
                f"polygon offsets must begin at zero and end at {payload_size} in "
                f"{reader.path}"
            )
        if np.any(np.diff(offsets) < 0):
            raise ValueError(f"polygon offsets are not monotone in {reader.path}")
        return offsets, connectivity, header_count - 1

    # In the classic layout, payload_size counts the leading size word of each
    # polygon as well as all vertex IDs.
    packed = np.asarray(reader.values(payload_size, "int"), dtype=np.int64)
    n_polygons = header_count
    offsets = np.empty(n_polygons + 1, dtype=np.int64)
    offsets[0] = 0
    connectivity_parts: list[IntArray] = []
    cursor = 0
    total_vertices = 0
    for polygon_index in range(n_polygons):
        if cursor >= packed.size:
            raise ValueError(
                f"classic polygon payload ends before polygon {polygon_index} in "
                f"{reader.path}"
            )
        size = int(packed[cursor])
        cursor += 1
        if size < 0 or cursor + size > packed.size:
            raise ValueError(
                f"invalid size {size} for polygon {polygon_index} in {reader.path}"
            )
        connectivity_parts.append(packed[cursor : cursor + size])
        cursor += size
        total_vertices += size
        offsets[polygon_index + 1] = total_vertices
    if cursor != packed.size:
        raise ValueError(
            f"classic polygon payload has {packed.size - cursor} unused integers in "
            f"{reader.path}"
        )
    connectivity = (
        np.concatenate(connectivity_parts)
        if connectivity_parts
        else np.empty(0, dtype=np.int64)
    )
    return offsets, connectivity, n_polygons


def _attribute_array(
    reader: _LegacyReader,
    directive: str,
    *,
    tuple_count: int,
) -> tuple[str, DataArray] | None:
    """Read one non-FIELD point/cell attribute directive."""

    tokens = directive.split()
    kind = tokens[0].upper()
    if kind == "SCALARS":
        if len(tokens) not in {3, 4}:
            raise ValueError(f"malformed SCALARS directive in {reader.path}")
        components = int(tokens[3]) if len(tokens) == 4 else 1
        if components <= 0:
            raise ValueError(f"SCALARS component count must be positive in {reader.path}")
        lookup = _parse_header(reader.nonempty_line() or "", "LOOKUP_TABLE", reader.path)
        if len(lookup) != 2:
            raise ValueError(f"malformed LOOKUP_TABLE directive in {reader.path}")
        values = reader.values(tuple_count * components, tokens[2])
        array = values.reshape(tuple_count, components)
        return tokens[1], array[:, 0] if components == 1 else array
    if kind in {"VECTORS", "NORMALS"}:
        if len(tokens) != 3:
            raise ValueError(f"malformed {kind} directive in {reader.path}")
        return tokens[1], reader.values(tuple_count * 3, tokens[2]).reshape(tuple_count, 3)
    if kind == "TENSORS":
        if len(tokens) != 3:
            raise ValueError(f"malformed TENSORS directive in {reader.path}")
        return tokens[1], reader.values(tuple_count * 9, tokens[2]).reshape(tuple_count, 3, 3)
    if kind == "TEXTURE_COORDINATES":
        if len(tokens) != 4:
            raise ValueError(
                f"malformed TEXTURE_COORDINATES directive in {reader.path}"
            )
        components = int(tokens[2])
        if components not in {1, 2, 3}:
            raise ValueError(
                f"TEXTURE_COORDINATES dimension must be 1, 2, or 3 in {reader.path}"
            )
        values = reader.values(tuple_count * components, tokens[3])
        array = values.reshape(tuple_count, components)
        return tokens[1], array[:, 0] if components == 1 else array
    return None


def _triangulate_polygons(
    points: FloatArray,
    offsets: IntArray,
    connectivity: IntArray,
) -> tuple[IntArray, IntArray, IntArray]:
    """Triangulate ordered polygons and return triangles and source metadata."""

    n_polygons = offsets.size - 1
    triangles: list[tuple[int, int, int]] = []
    parents: list[int] = []
    source_sizes: list[int] = []
    epsilon = np.finfo(float).eps
    for polygon_id in range(n_polygons):
        polygon = connectivity[offsets[polygon_id] : offsets[polygon_id + 1]]
        size = int(polygon.size)
        if size < 3:
            raise ValueError(f"polygon {polygon_id} has only {size} vertices")
        if np.unique(polygon).size != size:
            raise ValueError(f"polygon {polygon_id} repeats a vertex index")
        if size == 3:
            children = [(int(polygon[0]), int(polygon[1]), int(polygon[2]))]
        elif size == 4:
            a, b, c, d = (int(value) for value in polygon)
            diagonal_ac = float(np.linalg.norm(points[a] - points[c]))
            diagonal_bd = float(np.linalg.norm(points[b] - points[d]))
            tolerance = 64.0 * epsilon * max(diagonal_ac, diagonal_bd, 1.0)
            if diagonal_ac < diagonal_bd - tolerance:
                use_ac = True
            elif diagonal_bd < diagonal_ac - tolerance:
                use_ac = False
            else:
                use_ac = tuple(sorted((a, c))) <= tuple(sorted((b, d)))
            children = (
                [(a, b, c), (a, c, d)]
                if use_ac
                else [(a, b, d), (b, c, d)]
            )
        else:
            # The clinical files contain only triangles and quadrilaterals.
            # Retain deterministic support for a larger simple polygon by
            # preserving its listed orientation and using a fan at vertex 0.
            anchor = int(polygon[0])
            children = [
                (anchor, int(polygon[index]), int(polygon[index + 1]))
                for index in range(1, size - 1)
            ]
        triangles.extend(children)
        parents.extend([polygon_id] * len(children))
        source_sizes.extend([size] * len(children))
    return (
        np.asarray(triangles, dtype=np.int64),
        np.asarray(parents, dtype=np.int64),
        np.asarray(source_sizes, dtype=np.int16),
    )


def clean_triangular_surface(
    points: NDArray[np.generic],
    triangles: NDArray[np.generic],
    point_data: Mapping[str, NDArray[np.generic]],
    cell_data: Mapping[str, NDArray[np.generic]],
    *,
    area_tolerance: float | None = None,
) -> tuple[FloatArray, IntArray, dict[str, DataArray], dict[str, DataArray], SurfaceCleanupReport]:
    """Remove unusable triangles and compact vertices without merging them.

    Degenerate triangles are those with a repeated index or area at or below
    ``area_tolerance``.  When the tolerance is omitted it is scaled from the
    largest input edge using the same roundoff-level rule as
    :meth:`surface_mesh.TriangleMesh.quality_report`.  Duplicate triangles are
    compared by their unordered vertex triplets, and the first valid input
    occurrence is retained.  Finally, points unused by the retained triangles
    are removed in ascending original-index order.

    No coincident or near-coincident vertices are merged.  Every point and cell
    array is sliced with the same deterministic maps, and the report lists both
    removed triangle indices and their source polygon IDs when the provenance
    field is present.
    """

    xyz = np.asarray(points, dtype=np.float64)
    cells_raw = np.asarray(triangles)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 3:
        raise ValueError("points must have shape (N, 3) with N >= 3")
    if not np.isfinite(xyz).all():
        raise ValueError("point coordinates must be finite")
    if cells_raw.ndim != 2 or cells_raw.shape[1] != 3 or cells_raw.shape[0] == 0:
        raise ValueError("triangles must have shape (M, 3) with M >= 1")
    if not np.issubdtype(cells_raw.dtype, np.integer):
        raise ValueError("triangle connectivity must use an integer dtype")
    cells = np.asarray(cells_raw, dtype=np.int64)
    if int(cells.min()) < 0 or int(cells.max()) >= xyz.shape[0]:
        raise ValueError("triangle connectivity contains an out-of-range point index")

    checked_point_data: dict[str, DataArray] = {}
    for name, values in point_data.items():
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"point_data[{name!r}] must have first dimension {xyz.shape[0]}"
            )
        checked_point_data[str(name)] = array
    checked_cell_data: dict[str, DataArray] = {}
    for name, values in cell_data.items():
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != cells.shape[0]:
            raise ValueError(
                f"cell_data[{name!r}] must have first dimension {cells.shape[0]}"
            )
        checked_cell_data[str(name)] = array

    triangle_points = xyz[cells]
    edge_lengths = np.linalg.norm(
        np.stack(
            (
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 1],
                triangle_points[:, 0] - triangle_points[:, 2],
            ),
            axis=1,
        ),
        axis=2,
    )
    length_scale = max(float(np.max(edge_lengths)), np.finfo(float).tiny)
    if area_tolerance is None:
        area_tolerance = 128.0 * np.finfo(float).eps * length_scale**2
    if not np.isfinite(area_tolerance) or area_tolerance < 0.0:
        raise ValueError("area_tolerance must be finite and non-negative")
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangle_points[:, 1] - triangle_points[:, 0],
            triangle_points[:, 2] - triangle_points[:, 0],
        ),
        axis=1,
    )
    repeated_index = (
        (cells[:, 0] == cells[:, 1])
        | (cells[:, 1] == cells[:, 2])
        | (cells[:, 2] == cells[:, 0])
    )
    zero_area = repeated_index | (areas <= area_tolerance)

    keep = ~zero_area
    valid_indices = np.flatnonzero(keep)
    duplicate_indices: list[int] = []
    if valid_indices.size:
        canonical = np.sort(cells[valid_indices], axis=1)
        _, first_positions = np.unique(canonical, axis=0, return_index=True)
        first_mask = np.zeros(valid_indices.size, dtype=bool)
        first_mask[first_positions] = True
        duplicate_indices = valid_indices[~first_mask].astype(int).tolist()
        keep[valid_indices[~first_mask]] = False

    retained_indices = np.flatnonzero(keep)
    if retained_indices.size == 0:
        raise ValueError("cleanup would remove every triangle")
    retained_cells = cells[retained_indices]
    used_points = np.unique(retained_cells.ravel()).astype(np.int64)
    removed_points = np.setdiff1d(
        np.arange(xyz.shape[0], dtype=np.int64), used_points, assume_unique=True
    )
    remap = np.full(xyz.shape[0], -1, dtype=np.int64)
    remap[used_points] = np.arange(used_points.size, dtype=np.int64)

    removed_triangle_indices = np.flatnonzero(~keep)
    removed_source_ids: tuple[int, ...] = ()
    if "source_polygon_id" in checked_cell_data:
        source_ids = np.asarray(checked_cell_data["source_polygon_id"])
        if source_ids.ndim != 1 or not np.issubdtype(source_ids.dtype, np.integer):
            raise ValueError("cell_data['source_polygon_id'] must be a 1-D integer array")
        removed_source_ids = tuple(
            int(value) for value in np.unique(source_ids[removed_triangle_indices])
        )

    cleaned_point_data = {
        name: np.asarray(values[used_points]).copy()
        for name, values in checked_point_data.items()
    }
    cleaned_cell_data = {
        name: np.asarray(values[retained_indices]).copy()
        for name, values in checked_cell_data.items()
    }
    report = SurfaceCleanupReport(
        input_points=int(xyz.shape[0]),
        output_points=int(used_points.size),
        input_triangles=int(cells.shape[0]),
        output_triangles=int(retained_indices.size),
        area_tolerance=float(area_tolerance),
        removed_zero_area_triangle_indices=tuple(
            int(value) for value in np.flatnonzero(zero_area)
        ),
        removed_duplicate_triangle_indices=tuple(duplicate_indices),
        removed_source_polygon_ids=removed_source_ids,
        removed_isolated_point_indices=tuple(int(value) for value in removed_points),
    )
    return (
        np.ascontiguousarray(xyz[used_points], dtype=np.float64),
        np.ascontiguousarray(remap[retained_cells], dtype=np.int64),
        cleaned_point_data,
        cleaned_cell_data,
        report,
    )


def read_legacy_polydata(path: str | Path) -> PolyDataResult:
    """Read and triangulate a numeric legacy-VTK ``POLYDATA`` surface.

    Returns ``(points, triangles, point_data, cell_data)``.  Point coordinates
    have shape ``(N, 3)`` and native-endian ``float64`` dtype.  Attribute names
    and scalar/vector shape are retained.  Numeric attribute dtypes are the
    native-endian equivalents of the VTK types in the file.
    """

    mesh_path = Path(path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh file not found: {mesh_path}")
    with mesh_path.open("rb") as handle:
        first = handle.readline()
        if not first.startswith(b"# vtk DataFile Version"):
            raise ValueError(f"not a legacy VTK file: {mesh_path}")
        if handle.readline() == b"":
            raise ValueError(f"missing VTK title line in {mesh_path}")
        mode = handle.readline().decode("ascii", errors="strict").strip().upper()
        if mode not in {"ASCII", "BINARY"}:
            raise ValueError(f"unsupported VTK payload mode {mode!r} in {mesh_path}")
        reader = _LegacyReader(handle, binary=mode == "BINARY", path=mesh_path)
        dataset = _parse_header(reader.nonempty_line() or "", "DATASET", mesh_path)
        if len(dataset) != 2 or dataset[1].upper() != "POLYDATA":
            raise ValueError(f"expected DATASET POLYDATA in {mesh_path}")

        point_header = _parse_header(reader.nonempty_line() or "", "POINTS", mesh_path)
        if len(point_header) != 3:
            raise ValueError(f"malformed POINTS directive in {mesh_path}")
        n_points = _positive_count(point_header[1], "point count", mesh_path)
        points = np.asarray(
            reader.values(3 * n_points, point_header[2]), dtype=np.float64
        ).reshape(n_points, 3)
        if not np.isfinite(points).all():
            raise ValueError(f"point coordinates contain non-finite values in {mesh_path}")

        polygon_header = _parse_header(
            reader.nonempty_line() or "", "POLYGONS", mesh_path
        )
        if len(polygon_header) != 3:
            raise ValueError(f"malformed POLYGONS directive in {mesh_path}")
        polygon_header_count = _positive_count(
            polygon_header[1], "polygon/offset count", mesh_path
        )
        payload_size = _positive_count(
            polygon_header[2], "polygon payload size", mesh_path
        )
        offsets, connectivity, n_polygons = _decode_polygons(
            reader, header_count=polygon_header_count, payload_size=payload_size
        )
        if connectivity.size == 0:
            raise ValueError(f"surface contains no polygon connectivity in {mesh_path}")
        if int(connectivity.min()) < 0 or int(connectivity.max()) >= n_points:
            raise ValueError(f"polygon connectivity is outside [0, {n_points}) in {mesh_path}")

        point_data: dict[str, DataArray] = {}
        polygon_data: dict[str, DataArray] = {}
        active_data: dict[str, DataArray] | None = None
        active_count: int | None = None
        while True:
            directive = reader.nonempty_line(allow_eof=True)
            if directive is None:
                break
            tokens = directive.split()
            kind = tokens[0].upper()
            if kind in {"POINT_DATA", "CELL_DATA"}:
                if len(tokens) != 2:
                    raise ValueError(f"malformed {kind} directive in {mesh_path}")
                count = _positive_count(tokens[1], f"{kind} tuple count", mesh_path)
                expected = n_points if kind == "POINT_DATA" else n_polygons
                if count != expected:
                    raise ValueError(
                        f"{kind} declares {count} tuples, expected {expected} in {mesh_path}"
                    )
                active_data = point_data if kind == "POINT_DATA" else polygon_data
                active_count = count
                continue
            if active_data is None or active_count is None:
                raise ValueError(
                    f"attribute directive {directive!r} appears before POINT_DATA or "
                    f"CELL_DATA in {mesh_path}"
                )
            if kind == "FIELD":
                if len(tokens) != 3:
                    raise ValueError(f"malformed FIELD directive in {mesh_path}")
                array_count = _positive_count(tokens[2], "FIELD array count", mesh_path)
                for _ in range(array_count):
                    header = (reader.nonempty_line() or "").split()
                    if len(header) != 4:
                        raise ValueError(f"malformed FIELD array header in {mesh_path}")
                    name, component_token, tuple_token, vtk_type = header
                    components = _positive_count(
                        component_token, f"component count for {name}", mesh_path
                    )
                    tuples = _positive_count(tuple_token, f"tuple count for {name}", mesh_path)
                    if tuples != active_count:
                        raise ValueError(
                            f"attribute {name!r} has {tuples} tuples, expected "
                            f"{active_count} in {mesh_path}"
                        )
                    if name in active_data:
                        raise ValueError(f"duplicate attribute name {name!r} in {mesh_path}")
                    values = reader.values(components * tuples, vtk_type)
                    reshaped = values.reshape(tuples, components)
                    active_data[name] = reshaped[:, 0] if components == 1 else reshaped
                continue
            parsed = _attribute_array(
                reader, directive, tuple_count=active_count
            )
            if parsed is None:
                raise ValueError(f"unsupported attribute directive {directive!r} in {mesh_path}")
            name, values = parsed
            if name in active_data:
                raise ValueError(f"duplicate attribute name {name!r} in {mesh_path}")
            active_data[name] = values

    triangles, parents, source_sizes = _triangulate_polygons(
        points, offsets, connectivity
    )
    if "source_polygon_id" in polygon_data or "source_polygon_size" in polygon_data:
        raise ValueError(
            "input cell attributes collide with reserved triangulation provenance names"
        )
    cell_data = {
        name: np.asarray(values)[parents]
        for name, values in polygon_data.items()
    }
    cell_data["source_polygon_id"] = parents
    cell_data["source_polygon_size"] = source_sizes
    return (
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(triangles, dtype=np.int64),
        point_data,
        cell_data,
    )


__all__ = [
    "PolyDataResult",
    "SurfaceCleanupReport",
    "clean_triangular_surface",
    "read_legacy_polydata",
]
