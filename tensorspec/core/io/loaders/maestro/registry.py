from __future__ import annotations

import h5py

from tensorspec.core.io.loaders.maestro.detect import (
    assert_maestro_signature,
    is_fixed_mode,
    select_spectra_dataset,
)
from tensorspec.core.io.loaders.maestro.kinds import (
    defl_x_line_4d,
    fermi_defl_3d,
    focus_xy_fine_5d,
    xy_fine_4d,
)
from tensorspec.core.io.loaders.maestro.low_level_scan import parse_low_level_scan
from tensorspec.core.io.loaders.maestro.types import ScanPlan

KIND_MODULES = [
    focus_xy_fine_5d,
    defl_x_line_4d,
    fermi_defl_3d,
    xy_fine_4d,
]


def match_kind(plan: ScanPlan, is_fixed: bool):
    for module in KIND_MODULES:
        if module.match(plan, is_fixed):
            return module
    return None


def load_with_kind(f: h5py.File, path: str) -> dict:
    assert_maestro_signature(f)
    is_fixed = is_fixed_mode(f)
    dataset = select_spectra_dataset(f)

    headers = f.get("Headers")
    if headers is None or "Low_Level_Scan" not in headers:
        raise ValueError("Missing Headers/Low_Level_Scan.")

    plan = parse_low_level_scan(headers["Low_Level_Scan"][()])
    module = match_kind(plan, is_fixed)
    if module is None:
        raise ValueError(
            f"No Maestro kind matched mode={plan.mode_name!r}, is_fixed={is_fixed}."
        )
    return module.load(f, plan, dataset, path=path)
