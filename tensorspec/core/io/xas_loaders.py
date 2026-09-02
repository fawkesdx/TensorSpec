"""Load 1D XAS / XMCD spectra from CSV or whitespace-delimited text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tensorspec.core.data_models import TensorData

ENERGY_ALIASES = (
    "energy",
    "e",
    "ev",
    "hv",
    "photon_energy",
    "photonenergy",
    "binding_energy",
    "be",
)

INTENSITY_ALIASES = (
    "intensity",
    "signal",
    "absorption",
    "abs",
    "mu",
    "xmu",
    "norm",
    "normalized",
)

POL_TAGS = ("CP", "CM", "LH", "LV")
I0_ALIASES = ("I0", "I_0", "i0", "beam_current", "BeamCurrent", "current")


def _match_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    fold = {alias.casefold(): alias for alias in aliases}
    for col in columns:
        if col.casefold() in fold:
            return col
    return None


def _match_pol_columns(columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    col_fold = {c.casefold(): c for c in columns}
    for tag in POL_TAGS:
        if tag.casefold() in col_fold:
            out[tag] = col_fold[tag.casefold()]
    return out


def _numeric_columns(df: pd.DataFrame, skip: set[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in skip:
            continue
        try:
            pd.to_numeric(df[col], errors="raise")
            cols.append(col)
        except (TypeError, ValueError):
            continue
    return cols


def _coerce_i0(series: pd.Series) -> float | None:
    for value in series:
        if pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            match = pd.Series([str(value)]).str.extract(
                r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            ).iloc[0, 0]
            if pd.notna(match):
                return float(match)
    return None


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep=r"\s+", engine="python")


def load_xas_spectrum(path: Path | str) -> TensorData:
    """
    Load one CSV/TXT with energy + intensity, or energy + two polarization columns.

    Returns:
      - (energy,) single spectrum, or
      - (channel, energy) paired CP/CM or LH/LV cube in /processed-ready form.
    """
    path = Path(path)
    df = _read_table(path)
    if df.empty or len(df.columns) < 2:
        raise ValueError("XAS file must have at least energy and one intensity column.")

    columns = [str(c) for c in df.columns]
    energy_col = _match_column(columns, ENERGY_ALIASES)
    if energy_col is None:
        energy_col = columns[0]

    energy = np.asarray(pd.to_numeric(df[energy_col], errors="coerce"), dtype=float)
    if not np.all(np.isfinite(energy)):
        raise ValueError(f"Energy column '{energy_col}' contains non-numeric values.")
    if energy.size < 2:
        raise ValueError("Need at least two energy points.")

    pol_cols = _match_pol_columns(columns)
    metadata: dict[str, Any] = {
        "source": str(path),
        "loader": "xas_csv",
        "energy_column": energy_col,
    }

    i0_col = _match_column(columns, I0_ALIASES)
    if i0_col is not None:
        i0_val = _coerce_i0(df[i0_col])
        if i0_val is not None:
            metadata["I0"] = i0_val

    if len(pol_cols) >= 2:
        tags = [t for t in POL_TAGS if t in pol_cols][:2]
        if len(tags) < 2:
            tags = list(pol_cols.keys())[:2]
        values = np.stack(
            [
                np.asarray(pd.to_numeric(df[pol_cols[tags[0]]], errors="coerce"), dtype=float),
                np.asarray(pd.to_numeric(df[pol_cols[tags[1]]], errors="coerce"), dtype=float),
            ],
            axis=0,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Polarization intensity columns contain non-numeric values.")
        metadata["channel_tags"] = tags
        metadata["pair_mode"] = f"{tags[0]}_{tags[1]}"
        return TensorData(
            value=values,
            axes=[np.arange(2, dtype=float), energy],
            labels=["channel", "energy"],
            units=["", "eV"],
            data_type="Experimental XAS (paired)",
            metadata=metadata,
        )

    skip = {energy_col}
    skip.update(pol_cols.values())
    if i0_col:
        skip.add(i0_col)
    intensity_col = _match_column(columns, INTENSITY_ALIASES)
    if intensity_col is None:
        numeric = _numeric_columns(df, skip)
        if not numeric:
            raise ValueError("Could not find an intensity column.")
        intensity_col = numeric[0]

    intensity = np.asarray(pd.to_numeric(df[intensity_col], errors="coerce"), dtype=float)
    if not np.all(np.isfinite(intensity)):
        raise ValueError(f"Intensity column '{intensity_col}' contains non-numeric values.")
    metadata["intensity_column"] = intensity_col

    return TensorData(
        value=intensity,
        axes=[energy],
        labels=["energy"],
        units=["eV"],
        data_type="Experimental XAS",
        metadata=metadata,
    )


def load_xas_pair(
    plus_path: Path | str,
    minus_path: Path | str,
    *,
    tag_plus: str = "CP",
    tag_minus: str = "CM",
) -> TensorData:
    """Load two 1D files and stack as a paired (channel, energy) cube."""
    plus = load_xas_spectrum(plus_path)
    minus = load_xas_spectrum(minus_path)
    if plus.labels != ["energy"] or minus.labels != ["energy"]:
        raise ValueError("Pair load expects single-column intensity files.")
    e_plus = np.asarray(plus.axes[0], dtype=float)
    e_minus = np.asarray(minus.axes[0], dtype=float)
    if e_plus.shape != e_minus.shape or not np.allclose(e_plus, e_minus):
        raise ValueError("Plus and minus spectra must share the same energy axis.")

    values = np.stack([plus.value, minus.value], axis=0)
    meta = dict(plus.metadata or {})
    meta.update(
        {
            "source_plus": str(plus_path),
            "source_minus": str(minus_path),
            "loader": "xas_pair",
            "channel_tags": [tag_plus, tag_minus],
            "pair_mode": f"{tag_plus}_{tag_minus}",
        }
    )
    if minus.metadata.get("I0") is not None and "I0" not in meta:
        meta["I0"] = minus.metadata["I0"]

    return TensorData(
        value=values,
        axes=[np.arange(2, dtype=float), e_plus],
        labels=["channel", "energy"],
        units=["", "eV"],
        data_type="Experimental XAS (paired)",
        metadata=meta,
    )
