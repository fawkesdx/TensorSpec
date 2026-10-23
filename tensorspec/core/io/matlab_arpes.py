"""Save an ARPES tensor as a MATLAB struct named from the tensor's own labels.

``data.value`` is the intensity, in the tensor's axis order. Each axis is a
1×N field whose name is that label (``Energy``, ``kx_Slit``, ``Photon_Energy``).
``data.info`` is the metadata.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.io import savemat

from tensorspec.core.data_models import TensorData

_GREEK = {"Θ": "Theta", "Φ": "Phi", "θ": "theta", "φ": "phi"}


def is_arpes(tensor: TensorData) -> bool:
    kind = str(getattr(tensor, "data_type", "") or "").casefold()
    if "arpes" in kind or "maestro" in kind:
        return True
    blob = " ".join(str(lab) for lab in tensor.labels).casefold()
    return "energy" in blob and any(
        key in blob for key in ("slit", "defl", "kx", "ky", "kz", "theta", "phi", "photon", "angle")
    )


def matlab_field_name(label: str, used: set[str]) -> str:
    text = str(label)
    for src, dst in _GREEK.items():
        text = text.replace(src, dst)
    name = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    if not name or name[0].isdigit():
        name = "axis_" + name
    base = name
    n = 2
    while name in used:
        name = f"{base}_{n}"
        n += 1
    used.add(name)
    return name


def _info(metadata: dict) -> dict:
    raw = dict(metadata or {})
    raw.pop("axis_alternates", None)

    def conv(v):
        if v is None:
            return ""
        if isinstance(v, (bool, np.bool_)):
            return int(bool(v))
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            if not v:
                return np.zeros((0, 1))
            if all(isinstance(i, str) for i in v):
                return np.array(list(v), dtype=object)
            if all(
                isinstance(i, (int, float, np.integer, np.floating)) and not isinstance(i, bool)
                for i in v
            ):
                return np.asarray(v, float).reshape(1, -1)
            return np.array([str(i) for i in v], dtype=object)
        if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                return np.array([str(i) for i in v.ravel().tolist()], dtype=object)
            if v.ndim == 1:
                return np.asarray(v, float).reshape(1, -1)
            return np.asarray(v)
        return str(v)

    return {str(k): conv(v) for k, v in raw.items()}


def arpes_matlab_payload(tensor: TensorData) -> dict:
    if not is_arpes(tensor):
        raise ValueError("Save to MATLAB is for ARPES data.")
    value = np.asarray(tensor.value)
    if value.ndim != len(tensor.labels) or value.ndim != len(tensor.axes):
        raise ValueError("ARPES tensor labels do not match the intensity dimensions.")
    used = {"value", "info"}
    payload = {"value": np.ascontiguousarray(value), "info": _info(tensor.metadata)}
    for label, axis, n in zip(tensor.labels, tensor.axes, value.shape):
        coords = np.asarray(axis, float).ravel()
        if coords.size != n:
            raise ValueError(f"axis '{label}' has length {coords.size}, intensity has {n}.")
        payload[matlab_field_name(label, used)] = coords.reshape(1, -1)
    return payload


def save_arpes_matlab(tensor: TensorData, path: str) -> None:
    savemat(path, {"data": arpes_matlab_payload(tensor)}, do_compression=True)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
#: Field names that are never an axis.
_NON_AXIS_FIELDS = {"value", "info", "para", "data"}
#: Sandy's structs name the axes ``x`` / ``y`` / ``z`` in the order of ``value``'s dimensions.
_XYZ = ("x", "y", "z")


def _guess_unit(label: str) -> str:
    low = str(label).casefold()
    if low.startswith("k") or "momentum" in low:
        return "1/A"
    if "energy" in low or low in ("e", "hv", "photon_energy"):
        return "eV"
    if any(key in low for key in ("angle", "slit", "defl", "theta", "phi", "tilt", "polar", "azimuth")):
        return "deg"
    return ""


def _looks_k_converted(var: str | None, para: dict | None, x: np.ndarray) -> bool:
    """Whether the first axis of a ``value/x/y`` struct is momentum rather than angle."""
    if var and re.search(r"(^|_)k(_|$|x|y|z|sp)", str(var)):
        return True
    if isinstance(para, dict) and any(key in para for key in ("k_step", "latt_const", "kx", "ky")):
        return True
    finite = np.asarray(x, float)
    finite = finite[np.isfinite(finite)]
    # Analyser angles span tens of degrees; momenta a few inverse angstroms at most.
    return finite.size > 0 and float(np.nanmax(np.abs(finite))) < 4.0


def _xyz_labels(var, para, axes) -> tuple[list[str], list[str]]:
    """Labels and units for Sandy's ``value(x, y[, z])`` convention: x is momentum or slit
    angle, y is ``E - E_F``, z is the third scan axis."""
    k_space = _looks_k_converted(var, para, axes[0])
    labels = ["kx" if k_space else "Slit Angle", "Energy", "Deflection"]
    units = ["1/A" if k_space else "deg", "eV", "deg"]
    return labels[: len(axes)], units[: len(axes)]


# -- v7.3 (HDF5) --------------------------------------------------------------
def _h5_value(obj, file):
    """A MATLAB value from an h5py node: char arrays as str, numeric arrays in MATLAB's
    dimension order, structs as dicts, cells as lists; references are followed."""
    import h5py

    if isinstance(obj, h5py.Group):
        fields = obj.attrs.get("MATLAB_fields")
        names = ([("".join(c.decode() for c in f)) for f in fields]
                 if fields is not None else list(obj.keys()))
        return {name: _h5_value(obj[name], file) for name in names if name in obj}
    cls = obj.attrs.get("MATLAB_class", b"")
    cls = cls.decode() if isinstance(cls, bytes) else str(cls)
    if obj.attrs.get("MATLAB_empty", 0):
        return "" if cls == "char" else np.zeros(0)
    raw = obj[()]
    if cls == "char":
        return "".join(chr(int(c)) for c in np.asarray(raw).ravel(order="F")).rstrip("\x00")
    if cls == "cell" or (isinstance(raw, np.ndarray) and raw.dtype.kind == "O"):
        out = []
        for ref in np.asarray(raw).ravel():
            try:
                out.append(_h5_value(file[ref], file))
            except Exception:
                out.append(None)
        return out
    arr = np.asarray(raw)
    if cls == "logical":
        arr = arr.astype(bool)
    if arr.size == 1 and arr.dtype.kind in "fiub":
        return arr.ravel()[0].item()          # MATLAB scalars are 1x1 arrays
    # h5py returns MATLAB's column-major array with the dimensions reversed.
    return arr.T if arr.ndim > 1 else arr


def _h5_candidates(file) -> list[str]:
    import h5py

    return [k for k in file.keys() if k != "#refs#" and isinstance(file[k], h5py.Group)
            and "value" in file[k]]


def _read_v73(path: str, var: str | None) -> tuple[str, dict]:
    import h5py

    with h5py.File(path, "r") as file:
        if var is None:
            names = _h5_candidates(file)
            if len(names) != 1:
                raise ValueError(
                    f"{path} holds {len(names)} ARPES structs; pass var=... (one of {names[:20]})")
            var = names[0]
        if var not in file:
            raise KeyError(f"variable {var!r} is not in {path}")
        return var, _h5_value(file[var], file)


# -- v7 and older (scipy) -----------------------------------------------------
def _mat_value(obj):
    """Unwrap scipy's ``mat_struct`` / object arrays into dicts, lists, strings and arrays."""
    from scipy.io.matlab import mat_struct

    if isinstance(obj, mat_struct):
        return {name: _mat_value(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            items = [_mat_value(o) for o in obj.ravel()]
            return items[0] if obj.size == 1 else items
        if obj.dtype.kind in "US":
            return str(obj.ravel()[0]) if obj.size == 1 else [str(s) for s in obj.ravel()]
        if obj.size == 1 and obj.dtype.kind in "fiub":
            return obj.ravel()[0].item()
        return obj
    if isinstance(obj, (bytes, np.bytes_)):
        return obj.decode()
    return obj


def _read_v7(path: str, var: str | None) -> tuple[str, dict]:
    from scipy.io import loadmat

    raw = loadmat(path, squeeze_me=False, struct_as_record=False, mat_dtype=False)
    structs = {k: _mat_value(v) for k, v in raw.items() if not k.startswith("__")}
    candidates = [k for k, v in structs.items() if isinstance(v, dict) and "value" in v]
    if var is None:
        if len(candidates) != 1:
            raise ValueError(
                f"{path} holds {len(candidates)} ARPES structs; pass var=... (one of {candidates})")
        var = candidates[0]
    if var not in structs:
        raise KeyError(f"variable {var!r} is not in {path}")
    return var, structs[var]


def _is_hdf5(path: str) -> bool:
    """v7.3 files are HDF5 behind a 512-byte text header; h5py looks past that."""
    import h5py

    return bool(h5py.is_hdf5(path))


def _scalar(value, default=None):
    if value is None or isinstance(value, (str, dict)):
        return default
    try:
        arr = np.asarray(value, float).ravel()
    except (TypeError, ValueError):
        return default
    return float(arr[0]) if arr.size else default


def load_arpes_matlab(path: str, var: str | None = None, *, labels=None, units=None) -> TensorData:
    """Read one ARPES struct from a MATLAB file into a :class:`TensorData`.

    Two struct layouts are understood:

    * the one :func:`save_arpes_matlab` writes -- ``value`` in the tensor's axis order plus one
      axis field per label (``Energy``, ``kx_Slit`` ...) and ``info``;
    * Sandy's MATLAB convention -- ``value(x, y[, z])`` with axis vectors ``x`` / ``y`` / ``z``,
      ``info`` (beamline, analyser, temperatures ...) and optionally ``para`` (the k-conversion
      parameters). ``x`` becomes ``kx`` (1/A) when the variable name or ``para`` says the cut is
      k-converted, else ``Slit Angle`` (deg); ``y`` is ``Energy`` (eV), ``z`` ``Deflection``.

    v7.3 files are read with h5py (which returns MATLAB arrays with their dimensions reversed;
    they are transposed back), older ones with scipy. ``var`` is required when the file holds
    more than one struct with a ``value`` field. ``labels`` / ``units`` override the guesses.
    The struct's ``info`` and ``para`` go into ``metadata`` unchanged, next to
    ``matlab_variable`` and ``matlab_file``.
    """
    path = str(path)
    if _is_hdf5(path):
        var, struct = _read_v73(path, var)
    else:
        var, struct = _read_v7(path, var)
    if "value" not in struct:
        raise ValueError(f"{var!r} has no 'value' field")
    value = np.asarray(struct["value"], float)
    para = struct.get("para") if isinstance(struct.get("para"), dict) else None

    if all(k in struct for k in _XYZ[: value.ndim]):
        axes = [np.asarray(struct[k], float).ravel() for k in _XYZ[: value.ndim]]
        guess_labels, guess_units = _xyz_labels(var, para, axes)
    else:
        fields = [k for k in struct if k not in _NON_AXIS_FIELDS
                  and isinstance(struct[k], np.ndarray) and struct[k].dtype.kind in "fiu"]
        axes, guess_labels = [], []
        for n in value.shape:
            hit = next((k for k in fields if k not in guess_labels
                        and np.asarray(struct[k]).size == n), None)
            if hit is None:
                raise ValueError(f"{var!r}: no axis field of length {n} for value{value.shape}")
            axes.append(np.asarray(struct[hit], float).ravel())
            guess_labels.append(hit)
        guess_units = [_guess_unit(lab) for lab in guess_labels]

    if value.ndim == 2 and value.shape != tuple(a.size for a in axes) \
            and value.shape == tuple(a.size for a in axes)[::-1]:
        value = value.T                      # the writer's order disagreed with the axes
    if value.shape != tuple(a.size for a in axes):
        raise ValueError(f"{var!r}: value{value.shape} does not match axes "
                         f"{[a.size for a in axes]}")

    labels = list(labels) if labels else guess_labels
    units = list(units) if units else guess_units
    metadata = {"matlab_variable": var, "matlab_file": path}
    info = struct.get("info")
    if isinstance(info, dict):
        metadata["info"] = info
    if para is not None:
        metadata["para"] = para
    return TensorData(value=value, axes=axes, labels=labels, units=units,
                      data_type="Experimental ARPES", metadata=metadata)


def arpes_metadata_number(metadata: dict, *keys: str, default=None):
    """A number from nested ``info`` metadata, e.g. ``('info', 'Beamline', 'Eresolution')``."""
    node = metadata or {}
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return _scalar(node, default)
