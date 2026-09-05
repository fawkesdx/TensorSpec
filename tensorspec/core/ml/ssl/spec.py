"""SSL preprocess config dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

AxisRole = Literal["energy", "slit", "defl", "x", "y", "other"]
SampleModeName = Literal["fermi3d", "disp2d"]


@dataclass
class TrimSpec:
    ranges: dict[str, tuple[float, float]]  # role -> (lo, hi)
    source_kind: str
    note: str = ""


@dataclass
class NormSpec:
    clip_percentiles: tuple[float, float] = (1.0, 99.0)
    scope: Literal["per_sample", "per_file"] = "per_sample"
    dead_pixel_sigma: float = 6.0
    subsample_points: int = 512


@dataclass
class ResampleSpec:
    energy_size: int = 224
    slit_size: int = 224
    defl_size: int = 16  # used for fermi3d
    deg_per_raw_px: float | None = None  # None -> lookup table


@dataclass
class SampleSpec:
    mode: SampleModeName
    index_roles: tuple[str, ...]  # e.g. ("y", "x") or ("x",) or ()


@dataclass
class PreprocessConfig:
    """Config fragment stored in shard manifest (Plan A)."""

    trim: TrimSpec
    norm: NormSpec
    resample: ResampleSpec
    sample: SampleSpec
    seed: int = 0


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    return value


def to_jsonable(obj) -> dict:
    return _jsonable_value(asdict(obj))


def preprocess_config_from_dict(d: dict) -> PreprocessConfig:
    trim_d = d["trim"]
    norm_d = d["norm"]
    resample_d = d["resample"]
    sample_d = d["sample"]

    trim = TrimSpec(
        ranges={role: tuple(bounds) for role, bounds in trim_d["ranges"].items()},
        source_kind=trim_d["source_kind"],
        note=trim_d.get("note", ""),
    )
    norm = NormSpec(
        clip_percentiles=tuple(norm_d.get("clip_percentiles", (1.0, 99.0))),
        scope=norm_d.get("scope", "per_sample"),
        dead_pixel_sigma=norm_d.get("dead_pixel_sigma", 6.0),
        subsample_points=norm_d.get("subsample_points", 512),
    )
    resample = ResampleSpec(
        energy_size=resample_d.get("energy_size", 224),
        slit_size=resample_d.get("slit_size", 224),
        defl_size=resample_d.get("defl_size", 16),
        deg_per_raw_px=resample_d.get("deg_per_raw_px"),
    )
    sample = SampleSpec(
        mode=sample_d["mode"],
        index_roles=tuple(sample_d["index_roles"]),
    )
    return PreprocessConfig(
        trim=trim,
        norm=norm,
        resample=resample,
        sample=sample,
        seed=d.get("seed", 0),
    )
