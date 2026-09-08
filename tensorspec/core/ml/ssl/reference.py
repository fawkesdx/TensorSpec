from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_ROI_KEYS = (
    "source_id",
    "saved_utc",
    "reduce_mode",
    "display_labels",
    "dims",
)

_REQUIRED_INTEGRATION_LABELS = ("Energy", "Angle")


@dataclass(frozen=True)
class FloorReference:
    map: np.ndarray
    y_axis: np.ndarray | None
    x_axis: np.ndarray | None
    roi: dict[str, Any]


def build_roi_dict(
    *,
    labels: list[str],
    axes: list[np.ndarray],
    coords: dict[int, int],
    halfwidths: dict[int, int],
    reduce_mode: str,
    source_id: str,
    display_y_label: str,
    display_x_label: str,
    saved_utc: str,
) -> dict[str, Any]:
    display_indices = {labels.index(display_y_label), labels.index(display_x_label)}
    dims: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        if i in display_indices:
            continue
        axis = np.asarray(axes[i])
        c = int(coords[i])
        hw = int(halfwidths.get(i, 0))
        n = int(axis.shape[0])
        start = max(0, c - hw)
        end = min(n, c + hw + 1)
        dims.append(
            {
                "label": label,
                "unit": "",
                "center_index": c,
                "halfwidth_px": hw,
                "physical_lo": float(axis[start]),
                "physical_hi": float(axis[end - 1]),
            }
        )
    return {
        "source_id": source_id,
        "saved_utc": saved_utc,
        "reduce_mode": reduce_mode,
        "display_labels": [display_y_label, display_x_label],
        "dims": dims,
    }


def integrate_xy_map(
    value: np.ndarray,
    *,
    labels: list[str],
    coords: dict[int, int],
    halfwidths: dict[int, int],
    y_label: str,
    x_label: str,
    reduce_mode: str,
) -> np.ndarray:
    """Same semantics as SliceWidget._integration_slices + _reduce_hidden_axes."""
    for required in _REQUIRED_INTEGRATION_LABELS:
        if required not in labels:
            raise ValueError(f"missing required axis label: {required}")

    if y_label not in labels:
        raise ValueError(f"missing display axis label: {y_label}")
    if x_label not in labels:
        raise ValueError(f"missing display axis label: {x_label}")

    y_idx = labels.index(y_label)
    x_idx = labels.index(x_label)
    shape = value.shape
    slices: list[slice] = []
    for i in range(len(labels)):
        if i in (x_idx, y_idx):
            slices.append(slice(None))
            continue
        c = int(coords[i])
        hw = int(halfwidths.get(i, 0))
        n = int(shape[i])
        slices.append(slice(max(0, c - hw), min(n, c + hw + 1)))

    arr = value[tuple(slices)]
    reduce_axes = tuple(i for i in range(arr.ndim) if i not in (x_idx, y_idx))
    mode = reduce_mode.lower()
    if not reduce_axes:
        out = arr
    elif mode == "mean":
        out = np.mean(arr, axis=reduce_axes)
    else:
        out = np.sum(arr, axis=reduce_axes)

    out = np.asarray(out)
    if x_idx < y_idx:
        out = out.T
    return out


def save_floor_reference(path: str | Path, ref: FloorReference) -> None:
    roi_json = json.dumps(ref.roi, separators=(",", ":")).encode("utf-8")
    payload: dict[str, Any] = {
        "map": np.asarray(ref.map, dtype=np.float32),
        "roi_json": roi_json,
    }
    if ref.y_axis is not None:
        payload["y_axis"] = np.asarray(ref.y_axis)
    if ref.x_axis is not None:
        payload["x_axis"] = np.asarray(ref.x_axis)
    np.savez_compressed(path, **payload)


def load_floor_reference(path: str | Path) -> FloorReference:
    with np.load(path, allow_pickle=False) as data:
        map_ = np.asarray(data["map"])
        if map_.ndim != 2:
            raise ValueError("map must be 2D")
        if not np.all(np.isfinite(map_)):
            raise ValueError("map must be finite")

        roi_raw = data["roi_json"]
        if isinstance(roi_raw, np.ndarray):
            roi_raw = roi_raw.item()
        if isinstance(roi_raw, bytes):
            roi_text = roi_raw.decode("utf-8")
        else:
            roi_text = str(roi_raw)
        roi = json.loads(roi_text)

        missing = [key for key in REQUIRED_ROI_KEYS if key not in roi]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"roi missing required keys: {joined}")

        y_axis = np.asarray(data["y_axis"]) if "y_axis" in data else None
        x_axis = np.asarray(data["x_axis"]) if "x_axis" in data else None

    return FloorReference(
        map=map_.astype(np.float32, copy=False),
        y_axis=y_axis,
        x_axis=x_axis,
        roi=roi,
    )
