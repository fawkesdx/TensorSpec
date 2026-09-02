from __future__ import annotations

import h5py

from tensorspec.core.io.loaders.maestro.errors import MaestroSignatureError

_REQUIRED_GROUPS = ("0D_Data", "2D_Data", "Headers")


def assert_maestro_signature(f: h5py.File) -> None:
    missing = [name for name in _REQUIRED_GROUPS if name not in f]
    if missing:
        raise MaestroSignatureError(
            f"Not a MAESTRO file: missing {', '.join(missing)}."
        )


def select_spectra_dataset(f: h5py.File) -> h5py.Dataset:
    group = f["2D_Data"]
    if "Process_000" in group:
        return group["Process_000"]
    fixed_names = sorted(
        name for name in group.keys() if name.startswith("Fixed_Spectra")
    )
    if fixed_names:
        item = group[fixed_names[0]]
        if isinstance(item, h5py.Dataset):
            return item
    for name in group.keys():
        item = group[name]
        if isinstance(item, h5py.Dataset):
            return item
    raise ValueError("No spectra dataset found in 2D_Data.")


def is_fixed_mode(f: h5py.File) -> bool:
    headers = f.get("Headers")
    if headers is None:
        return False
    return "DAQ_Fixed" in headers
