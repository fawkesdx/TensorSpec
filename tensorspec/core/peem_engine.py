from __future__ import annotations

from typing import Any, Literal

import numpy as np

from tensorspec.core.data_models import TensorData

PairMode = Literal["auto", "CP_CM", "LH_LV"]

_VALID_MODES = frozenset({"auto", "CP_CM", "LH_LV"})

_PASSTHROUGH_KEYS = (
    "csv_attached",
    "I0",
    "source",
    "loader",
    "beamline_csv",
    "beamline_table",
)


def resolve_pair_mode(pol: list[str], mode: PairMode) -> tuple[str, list[str]]:
    """
    Returns (resolved_mode, channel_tags).
    auto: CP/CM-only → ("CP_CM", ["CP","CM"]); LH/LV-only → ("LH_LV", ["LH","LV"]);
    mixed or none → ValueError.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported pair mode: {mode!r}")

    if mode == "CP_CM":
        return "CP_CM", ["CP", "CM"]
    if mode == "LH_LV":
        return "LH_LV", ["LH", "LV"]

    has_cp_cm = any(tag in ("CP", "CM") for tag in pol)
    has_lh_lv = any(tag in ("LH", "LV") for tag in pol)

    if has_cp_cm and not has_lh_lv:
        return "CP_CM", ["CP", "CM"]
    if has_lh_lv and not has_cp_cm:
        return "LH_LV", ["LH", "LV"]
    raise ValueError("Cannot auto-resolve pair mode from pol tags")


def _validate_raw(tensor: TensorData) -> list[str]:
    if tensor.ndim != 3 or tensor.labels != ["frame", "y", "x"]:
        raise ValueError("Expected raw PEEM tensor with shape (frame, y, x)")

    pol = tensor.metadata.get("pol")
    if not isinstance(pol, list):
        raise ValueError("metadata['pol'] must be a list")

    n_frames = tensor.value.shape[0]
    if len(pol) < n_frames:
        raise ValueError("metadata['pol'] shorter than frame count")

    return list(pol[:n_frames])


def _frame_names(tensor: TensorData, n_frames: int) -> list[str | None]:
    names = tensor.metadata.get("frame_names")
    if not isinstance(names, list):
        return [None] * n_frames
    out: list[str | None] = []
    for i in range(n_frames):
        name = names[i] if i < len(names) else None
        out.append(name if isinstance(name, str) else None)
    return out


def _pair_indices(
    pol: list[str],
    channel_tags: list[str],
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    queues: dict[str, list[int]] = {tag: [] for tag in channel_tags}
    unpaired: list[dict[str, Any]] = []

    for idx, tag in enumerate(pol):
        if tag in queues:
            queues[tag].append(idx)
        else:
            unpaired.append({"frame_index": idx, "frame_name": None, "pol": tag})

    pairs: list[tuple[int, int]] = []
    tag_a, tag_b = channel_tags
    while queues[tag_a] and queues[tag_b]:
        pairs.append((queues[tag_a].pop(0), queues[tag_b].pop(0)))

    for tag in channel_tags:
        for idx in queues[tag]:
            unpaired.append({"frame_index": idx, "frame_name": None, "pol": tag})

    return pairs, unpaired


def pair_stack(tensor: TensorData, mode: PairMode) -> TensorData:
    """
    Input shape (frame,y,x) + metadata['pol'] (+ optional frame_names).
    Output shape (n_pairs, 2, y, x), data_type "Experimental PEEM (paired)".
    Metadata: pair_mode, channel_tags, pair_sources, unpaired, plus CSV/I0 passthrough.
    """
    pol = _validate_raw(tensor)
    pair_mode, channel_tags = resolve_pair_mode(pol, mode)
    n_frames, y_size, x_size = tensor.value.shape
    frame_names = _frame_names(tensor, n_frames)

    pairs, unpaired = _pair_indices(pol, channel_tags)
    if not pairs:
        raise ValueError("No pairs could be formed for the selected mode")

    for entry in unpaired:
        idx = entry["frame_index"]
        entry["frame_name"] = frame_names[idx]

    pair_sources: list[dict[str, Any]] = []
    stacked = np.empty((len(pairs), 2, y_size, x_size), dtype=float)
    for pair_idx, (idx_a, idx_b) in enumerate(pairs):
        stacked[pair_idx, 0] = tensor.value[idx_a]
        stacked[pair_idx, 1] = tensor.value[idx_b]
        pair_sources.append(
            {
                "pair": pair_idx,
                "channels": [
                    {
                        "tag": channel_tags[0],
                        "frame_index": idx_a,
                        "frame_name": frame_names[idx_a],
                    },
                    {
                        "tag": channel_tags[1],
                        "frame_index": idx_b,
                        "frame_name": frame_names[idx_b],
                    },
                ],
            }
        )

    metadata: dict[str, Any] = {
        "pair_mode": pair_mode,
        "channel_tags": list(channel_tags),
        "pair_sources": pair_sources,
        "unpaired": unpaired,
    }
    for key in _PASSTHROUGH_KEYS:
        if key in tensor.metadata:
            metadata[key] = tensor.metadata[key]

    return TensorData(
        value=stacked,
        axes=[np.arange(len(pairs)), np.arange(2), tensor.axes[1], tensor.axes[2]],
        labels=["pair", "channel", "y", "x"],
        units=["", "", tensor.units[1], tensor.units[2]],
        data_type="Experimental PEEM (paired)",
        metadata=metadata,
    )
