from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py

from tensorspec.core.io.loaders.maestro import registry
from tensorspec.core.io.loaders.maestro.detect import (
    assert_maestro_signature,
    is_fixed_mode,
    select_spectra_dataset,
)
from tensorspec.core.io.loaders.maestro.errors import MaestroSignatureError
from tensorspec.core.io.loaders.maestro.kinds import process000_generic
from tensorspec.core.io.loaders.maestro.low_level_scan import parse_low_level_scan
from tensorspec.core.io.loaders.maestro.types import ScanLoop, ScanMotor, ScanPlan


class MaestroLoader:
    def __init__(self, filepath: str):
        self.filepath = str(filepath)

    def load(self) -> dict:
        with h5py.File(self.filepath, "r") as f:
            assert_maestro_signature(f)
            headers = f["Headers"]
            if "Low_Level_Scan" in headers:
                plan = parse_low_level_scan(headers["Low_Level_Scan"][()])
                fixed = is_fixed_mode(f)
                module = registry.match_kind(plan, fixed)
                if module is None:
                    dataset = select_spectra_dataset(f)
                    if not fixed and dataset.name.endswith("/Process_000"):
                        return process000_generic.load(
                            f, dataset, path=self.filepath
                        )
                    actual = _actual_cycles(dataset.shape, plan.expected_cycles)
                    raise ValueError(
                        "No Maestro kind matched "
                        f"mode={plan.mode_name!r}, is_fixed={fixed}, "
                        f"expected_cycles={plan.expected_cycles}, "
                        f"actual_cycles={actual}."
                    )
                return registry.load_with_kind(f, self.filepath)

            process = f["2D_Data"].get("Process_000")
            if isinstance(process, h5py.Dataset):
                return process000_generic.load(f, process, path=self.filepath)

            raise ValueError(
                "MAESTRO file has neither Headers/Low_Level_Scan nor "
                "2D_Data/Process_000."
            )


def _actual_cycles(shape: tuple[int, ...], expected: int) -> int:
    if not shape:
        return 0
    return int(min(shape, key=lambda size: abs(int(size) - expected)))


__all__ = [
    "MaestroLoader",
    "MaestroSignatureError",
    "ScanLoop",
    "ScanMotor",
    "ScanPlan",
    "parse_low_level_scan",
]
