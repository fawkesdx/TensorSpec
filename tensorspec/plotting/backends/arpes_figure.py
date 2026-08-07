"""Headless ARPES figure export (heatmap + profile curves).

Builds the same layout the Qt SliceWidget saves: main plane with optional
side profiles. No GUI toolkit imports — matplotlib Agg only.
"""
from __future__ import annotations

import io
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def export_slice_figure(
    plane: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    x_profile: np.ndarray | None = None,
    y_profile: np.ndarray | None = None,
    x_label: str = "X",
    y_label: str = "Y",
    x_unit: str = "",
    y_unit: str = "",
    crosshair: tuple[float, float] | None = None,
    title: str = "",
    fmt: Literal["pdf", "svg"] = "pdf",
) -> bytes:
    """Render a 2D intensity map (and optional profiles) to PDF or SVG bytes."""
    show_profiles = x_profile is not None and y_profile is not None
    if show_profiles:
        figure = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, width_ratios=(4, 1.2), height_ratios=(4, 1.2))
        ax_main = figure.add_subplot(grid[0, 0])
        ax_y = figure.add_subplot(grid[0, 1], sharey=ax_main)
        ax_x = figure.add_subplot(grid[1, 0], sharex=ax_main)
    else:
        figure, ax_main = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
        ax_x = ax_y = None

    extent = [float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])]
    ax_main.imshow(
        plane,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
    )
    xlabel = f"{x_label} ({x_unit})" if x_unit else x_label
    ylabel = f"{y_label} ({y_unit})" if y_unit else y_label
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    if title:
        ax_main.set_title(title)

    if crosshair is not None:
        ax_main.axvline(crosshair[0], color="cyan", lw=0.8, alpha=0.8)
        ax_main.axhline(crosshair[1], color="cyan", lw=0.8, alpha=0.8)

    if show_profiles:
        ax_x.plot(x_axis, x_profile, color="#60a5fa", lw=1.2)
        ax_x.set_xlabel(xlabel)
        ax_x.set_ylabel("I")
        ax_y.plot(y_profile, y_axis, color="#f87171", lw=1.2)
        ax_y.set_xlabel("I")
        if crosshair is not None:
            ax_x.axvline(crosshair[0], color="cyan", lw=0.8, alpha=0.7)
            ax_y.axhline(crosshair[1], color="cyan", lw=0.8, alpha=0.7)

    buffer = io.BytesIO()
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    figure.savefig(buffer, format=fmt, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(figure)
    return buffer.getvalue()
