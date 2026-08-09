"""Headless Crystal structure figure export (atoms / bonds / cell).

No GUI toolkit imports — matplotlib Agg only. Flat materials (no PBR).
"""
from __future__ import annotations

import io
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

_DEFAULT_CPK = {
    "H": "#ffffff",
    "C": "#909090",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "Si": "#f0c8a0",
    "S": "#ffff30",
    "Fe": "#e06633",
    "Cu": "#c88033",
}
_FALLBACK = "#808080"

# Scatter area scale (pt²): marker size ≈ (radius * atom_scale * k)²
_MARKER_K = 50.0

_CELL_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 6),
    (3, 5),
    (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
)


def _atom_color(element: str, colors: dict[str, str] | None) -> str:
    palette = colors if colors is not None else _DEFAULT_CPK
    return palette.get(element, _FALLBACK)


def _marker_size(radius: float, atom_scale: float) -> float:
    return (radius * atom_scale * _MARKER_K) ** 2


def _cell_corners(cell: list[list[float]]) -> list[np.ndarray]:
    a, b, c = (np.asarray(v, dtype=float) for v in cell)
    origin = np.zeros(3)
    return [
        origin,
        a,
        b,
        c,
        a + b,
        a + c,
        b + c,
        a + b + c,
    ]


def _cell_segments(cell: list[list[float]]) -> list[list[np.ndarray]]:
    corners = _cell_corners(cell)
    return [[corners[i], corners[j]] for i, j in _CELL_EDGES]


def _camera_elev_azim(camera: dict) -> tuple[float, float]:
    position = np.asarray(camera["position"], dtype=float)
    target = np.asarray(camera["target"], dtype=float)
    direction = position - target
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return 20.0, 45.0
    direction /= norm
    r_xy = float(np.hypot(direction[0], direction[1]))
    elev = float(np.degrees(np.arctan2(direction[2], r_xy)))
    azim = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return elev, azim


def _set_equal_aspect(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-9] = 1.0
    center = (mins + maxs) / 2.0
    half = ranges / 2.0
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])
    ax.set_box_aspect(ranges)


def export_crystal_figure(
    *,
    atoms: list[dict],
    bonds: list[tuple[int, int]],
    cell: list[list[float]] | None,
    polyhedra: list[dict] | None = None,
    show_bonds: bool = True,
    show_cell: bool = True,
    atom_scale: float = 0.5,
    colors: dict[str, str] | None = None,
    title: str = "",
    fmt: Literal["png", "svg", "pdf"] = "png",
    camera: dict | None = None,
    dpi: int = 200,
) -> bytes:
    """Render atoms, bonds, and optional cell to PNG/SVG/PDF bytes."""
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    if atoms:
        positions = np.array([atom["position"] for atom in atoms], dtype=float)
        xs, ys, zs = positions.T
        atom_colors = [_atom_color(atom["element"], colors) for atom in atoms]
        sizes = [_marker_size(float(atom["radius"]), atom_scale) for atom in atoms]
        ax.scatter(xs, ys, zs, s=sizes, c=atom_colors, depthshade=True, edgecolors="none")
    else:
        positions = np.empty((0, 3))

    if show_bonds and bonds and atoms:
        segments = []
        for i, j in bonds:
            if 0 <= i < len(atoms) and 0 <= j < len(atoms):
                pi = np.asarray(atoms[i]["position"], dtype=float)
                pj = np.asarray(atoms[j]["position"], dtype=float)
                segments.append([pi, pj])
        if segments:
            bond_lines = Line3DCollection(segments, colors="#666666", linewidths=1.0)
            ax.add_collection3d(bond_lines)

    aspect_points = [positions] if len(positions) else []
    if show_cell and cell is not None:
        cell_segments = _cell_segments(cell)
        cell_lines = Line3DCollection(cell_segments, colors="#333333", linewidths=1.2)
        ax.add_collection3d(cell_lines)
        aspect_points.append(np.array([pt for seg in cell_segments for pt in seg]))

    if polyhedra:
        for polyhedron in polyhedra:
            vertices = np.asarray(polyhedron["vertices"], dtype=float)
            simplices = polyhedron["simplices"]
            faces = [[vertices[idx] for idx in simplex] for simplex in simplices]
            if not faces:
                continue
            face_collection = Poly3DCollection(
                faces,
                alpha=0.25,
                facecolor="#4fc3f7",
                edgecolor="#1565c0",
                linewidths=0.5,
            )
            ax.add_collection3d(face_collection)
            aspect_points.append(vertices)

    if aspect_points:
        _set_equal_aspect(ax, np.vstack(aspect_points))
    else:
        ax.set_box_aspect((1.0, 1.0, 1.0))

    if camera is not None:
        elev, azim = _camera_elev_azim(camera)
        ax.view_init(elev=elev, azim=azim)
    else:
        ax.view_init(elev=20, azim=45)

    if title:
        ax.set_title(title)

    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format=fmt,
        dpi=dpi if fmt == "png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return buf.getvalue()
