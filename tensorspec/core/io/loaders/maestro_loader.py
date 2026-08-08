"""MAESTRO (ALS BL7.0.2) HDF5 loader for current beamline layouts.

Supports Fixed and Swept analyzer modes. Intensity may live under
``2D_Data/*`` (cuts / focus scans) or ``Preview/*`` (XY spatial maps when
``2D_Data`` is empty). Detector axes come from ``scaleOffset`` /
``scaleDelta`` / ``unitNames``; scan motors prefer ``Low_Level_Scan`` when
point counts match array dimensions.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _clean_header_value(value: Any) -> Any:
    text = _as_str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    text = text.strip()
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", text):
            return float(text)
    except ValueError:
        pass
    return text


def _decode_attr_list(values) -> List[Any]:
    if values is None:
        return []
    out = []
    for item in np.atleast_1d(values):
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="replace"))
        elif isinstance(item, np.generic):
            out.append(item.item())
        else:
            out.append(item)
    return out


def read_measurement_log(csv_path: str | Path) -> Dict[str, Dict[str, str]]:
    """Index ``MAESTRO_Measurement_Log.csv`` rows by file name."""
    path = Path(csv_path)
    if not path.is_file():
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("File Name") or "").strip()
            if name:
                rows[name] = {k: (v or "").strip() for k, v in row.items()}
    return rows


def lookup_measurement_log(
    filepath: str | Path,
    log_path: str | Path | None = None,
) -> Optional[Dict[str, str]]:
    """Find the CSV row for ``filepath``, searching a sibling log by default."""
    path = Path(filepath)
    candidates = []
    if log_path:
        candidates.append(Path(log_path))
    candidates.append(path.parent / "MAESTRO_Measurement_Log.csv")
    for candidate in candidates:
        rows = read_measurement_log(candidate)
        if path.name in rows:
            return rows[path.name]
    return None


class MaestroLoader:
    """Parse a MAESTRO ``.h5`` file into a TensorSpec-ready payload."""

    SPECTRA_NAME = re.compile(r"^(Fixed|Swept)_Spectra", re.IGNORECASE)

    # Prefer these 0D channels when binding an extra data dimension.
    SCAN_AXIS_PRIORITY = (
        "Slit Defl",
        "mono_eV",
        "Sample X",
        "Sample Y",
        "Scan X",
        "Scan Y",
        "Optics Stage",
        "Mono EnergyDAQ",
        "Beamline EnergyDAQ",
        "EPU EnergyDAQ",
    )
    SCAN_AXIS_AVOID = ("cryostat", "time", "null", "nanoarpes", "num_sw")

    MOTOR_DISPLAY = {
        "mono_eV": ("Photon Energy", "eV"),
        "Mono EnergyDAQ": ("Photon Energy", "eV"),
        "Beamline EnergyDAQ": ("Photon Energy", "eV"),
        "EPU EnergyDAQ": ("Photon Energy", "eV"),
        "Slit Defl": ("Slit Deflection", "deg"),
    }

    def __init__(
        self,
        filepath,
        measurement_log_row: Optional[Dict[str, str]] = None,
        measurement_log_path: str | Path | None = None,
    ):
        self.filepath = str(filepath)
        self.filename = os.path.basename(self.filepath)
        self.measurement_log_row = measurement_log_row
        self.measurement_log_path = measurement_log_path

    def load(self) -> Dict[str, Any]:
        with h5py.File(self.filepath, "r") as handle:
            if "Headers" not in handle:
                raise ValueError("Not a MAESTRO file: Missing Headers group.")
            if "0D_Data" not in handle and "2D_Data" not in handle and "Preview" not in handle:
                raise ValueError("Not a MAESTRO file: Missing data groups.")

            headers = self._parse_all_headers(handle)
            is_fixed = "DAQ_Fixed" in handle["Headers"]
            if not is_fixed and "DAQ_Swept" not in handle["Headers"]:
                raise ValueError("Not a MAESTRO file: Missing DAQ_Fixed/DAQ_Swept.")

            daq = headers.get("DAQ_Fixed" if is_fixed else "DAQ_Swept", {})
            mode_string = self._mode_from_daq(daq, is_fixed)
            comments = self._parse_comments(handle)
            motors = self._load_motors(handle)
            scan_motors = self._resolve_scan_motors(
                self._parse_low_level_motors(headers.get("Low_Level_Scan", {})),
                motors,
            )

            ds, source_path = self._select_intensity_dataset(handle)
            raw = np.asarray(ds[()], dtype=float)
            unit_names = _decode_attr_list(ds.attrs.get("unitNames"))
            offsets = np.asarray(ds.attrs.get("scaleOffset", []), dtype=float)
            deltas = np.asarray(ds.attrs.get("scaleDelta", []), dtype=float)

            data, axes, labels, units = self._build_axes(
                raw,
                unit_names=unit_names,
                offsets=offsets,
                deltas=deltas,
                scan_motors=scan_motors,
                motors=motors,
            )

            log_row = self.measurement_log_row
            if log_row is None:
                log_row = lookup_measurement_log(self.filepath, self.measurement_log_path)

            measurement_kind = None
            if log_row:
                measurement_kind = log_row.get("Measurement Type") or None
                if measurement_kind:
                    mode_string = measurement_kind

            metadata = self._build_metadata(
                headers=headers,
                comments=comments,
                daq=daq,
                is_fixed=is_fixed,
                source_path=source_path,
                scan_motors=scan_motors,
                motors=motors,
                log_row=log_row,
                shape=tuple(int(n) for n in data.shape),
            )

        return {
            "name": self.filename.replace(".h5", "").replace(".H5", ""),
            "data": data,
            "axes": dict(zip(labels, axes)),
            "axis_units": units,
            "mode": mode_string,
            "is_fixed": is_fixed,
            "facility": "MAESTRO",
            "measurement_kind": measurement_kind,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------ parsers

    def _parse_all_headers(self, handle: h5py.File) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name in handle["Headers"].keys():
            table: Dict[str, Any] = {}
            for row in handle["Headers"][name][()]:
                key = _as_str(row["name"] if "name" in row.dtype.names else row[0]).strip()
                val = row["value"] if "value" in row.dtype.names else row[1]
                table[key] = _clean_header_value(val)
            out[name] = table
        return out

    def _parse_comments(self, handle: h5py.File) -> Dict[str, List[str]]:
        if "Comments" not in handle:
            return {}
        out: Dict[str, List[str]] = {}
        for name in handle["Comments"].keys():
            entries = []
            raw = handle["Comments"][name][()]
            for row in np.atleast_1d(raw):
                if hasattr(row, "dtype") and row.dtype.names:
                    parts = [_as_str(row[field]).strip() for field in row.dtype.names]
                    entries.append(" | ".join(p for p in parts if p))
                else:
                    entries.append(_as_str(row).strip())
            out[name] = entries
        return out

    def _load_motors(self, handle: h5py.File) -> Dict[str, np.ndarray]:
        if "0D_Data" not in handle:
            return {}
        motors = {}
        for name in handle["0D_Data"].keys():
            if self.SPECTRA_NAME.match(name) or name.endswith("_sum") or name.endswith("_num_sw_actual"):
                continue
            motors[name] = np.asarray(handle["0D_Data"][name][()], dtype=float).ravel()
        return motors

    def _parse_low_level_motors(self, low: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not low:
            return []
        try:
            n_motors = int(low.get("NMSBDV0", 0) or 0)
        except (TypeError, ValueError):
            n_motors = 0
        motors = []
        for index in range(n_motors):
            name = _as_str(low.get(f"NM_0_{index}", f"Motor_{index}")).strip("'\"")
            if not name or name.lower() in {"null", "none"}:
                continue
            unit = _as_str(low.get(f"UN_0_{index}", "")).strip("'\"")
            planned = low.get(f"N_0_{index}")
            if planned in (None, "", "None"):
                planned = low.get(f"NMPOS_{index}", low.get("NMPOS_0"))
            start = low.get(f"ST_0_{index}")
            end = low.get(f"EN_0_{index}")

            axis = None
            count = None
            try:
                if planned not in (None, "", "None"):
                    count = int(planned)
            except (TypeError, ValueError):
                count = None

            try:
                if start not in (None, "", "None") and end not in (None, "", "None") and count and count >= 1:
                    start_f = float(start)
                    end_f = float(end)
                    axis = (
                        np.linspace(start_f, end_f, count)
                        if count > 1
                        else np.array([start_f], dtype=float)
                    )
            except (TypeError, ValueError):
                axis = None

            motors.append(
                {
                    "name": name,
                    "unit": unit or self._unit_from_label(name),
                    "axis": axis,
                    "n": count,
                    "planned_n": count,
                }
            )
        return motors

    def _resolve_scan_motors(
        self,
        parsed: List[Dict[str, Any]],
        motors: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Bind Low_Level motor names to actual 0D traces (handles hv scans & aborts)."""
        resolved = []
        for motor in parsed:
            raw_name = motor["name"]
            arr = self._lookup_motor_array(raw_name, motors)
            display, unit = self._display_motor(raw_name, motor.get("unit"))
            planned = motor.get("planned_n")
            planned_axis = motor.get("axis")

            if arr is not None and arr.size > 0:
                # XY rasters store a flattened path in 0D that is longer than the
                # grid axis from ST/EN/N — keep the Low_Level grid in that case.
                if planned and planned_axis is not None and arr.size > int(planned):
                    axis = np.asarray(planned_axis, dtype=float)
                    count = int(planned)
                    aborted = False
                elif planned and arr.size < int(planned):
                    axis = np.asarray(arr, dtype=float)
                    count = int(arr.size)
                    aborted = True
                else:
                    axis = np.asarray(arr, dtype=float)
                    count = int(arr.size)
                    aborted = bool(planned) and count < int(planned)
                resolved.append(
                    {
                        "name": display,
                        "raw_name": raw_name,
                        "unit": unit,
                        "axis": axis,
                        "n": count,
                        "planned_n": planned,
                        "aborted": aborted,
                    }
                )
                continue

            if planned_axis is not None and motor.get("n"):
                resolved.append(
                    {
                        "name": display,
                        "raw_name": raw_name,
                        "unit": unit,
                        "axis": np.asarray(planned_axis, dtype=float),
                        "n": int(motor["n"]),
                        "planned_n": planned,
                        "aborted": False,
                    }
                )
                continue

            resolved.append(
                {
                    "name": display,
                    "raw_name": raw_name,
                    "unit": unit,
                    "axis": None,
                    "n": planned,
                    "planned_n": planned,
                    "aborted": False,
                }
            )
        return resolved

    def _lookup_motor_array(
        self, raw_name: str, motors: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        if raw_name in motors:
            return motors[raw_name]
        aliases = {
            "mono_eV": ("mono_eV", "Mono EnergyDAQ", "Beamline EnergyDAQ", "EPU EnergyDAQ"),
            "Slit Defl": ("Slit Defl",),
        }
        for candidate in aliases.get(raw_name, ()):
            if candidate in motors:
                return motors[candidate]
        # Case-insensitive fallback
        lower = raw_name.lower()
        for key, arr in motors.items():
            if key.lower() == lower:
                return arr
        return None

    def _display_motor(self, raw_name: str, unit: Optional[str] = None) -> Tuple[str, str]:
        if raw_name in self.MOTOR_DISPLAY:
            label, default_unit = self.MOTOR_DISPLAY[raw_name]
            return label, unit or default_unit
        return raw_name, unit or self._unit_from_label(raw_name)

    def _mode_from_daq(self, daq: Dict[str, Any], is_fixed: bool) -> str:
        lens = daq.get("SFLNM0") or daq.get("SSLNM0") or ""
        region = daq.get("SFRGN0") or daq.get("SSRGN0") or ""
        pass_energy = daq.get("SFPE_0") or daq.get("SSPE_0")
        bits = ["Fixed" if is_fixed else "Swept"]
        if region:
            bits.append(str(region))
        if lens:
            bits.append(str(lens))
        if pass_energy is not None and pass_energy != "":
            bits.append(f"PE={pass_energy}")
        return " / ".join(bits)

    def _select_intensity_dataset(self, handle: h5py.File) -> Tuple[h5py.Dataset, str]:
        for group_name in ("2D_Data", "Preview"):
            if group_name not in handle:
                continue
            group = handle[group_name]
            names = list(group.keys())
            preferred = [n for n in names if self.SPECTRA_NAME.match(n) and not n.endswith("_sum")]
            candidates = preferred or [n for n in names if not n.endswith("_sum")]
            if not candidates:
                continue
            name = sorted(candidates)[0]
            return group[name], f"{group_name}/{name}"
        raise ValueError("MAESTRO file has no intensity dataset under 2D_Data or Preview.")

    # ------------------------------------------------------------------ axes

    def _build_axes(
        self,
        raw: np.ndarray,
        *,
        unit_names: List[Any],
        offsets: np.ndarray,
        deltas: np.ndarray,
        scan_motors: List[Dict[str, Any]],
        motors: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray], List[str], List[str]]:
        data = np.squeeze(raw)
        if data.ndim == 0:
            data = data.reshape((1,))

        n_scale = int(min(len(offsets), len(deltas), len(unit_names), data.ndim))
        used_scan = set()

        # Spatial maps often store motors as (Ny, Nx) while unitNames follow
        # Low_Level order (X, Y). Prefer scan-motor lengths when every dim matches.
        if (
            scan_motors
            and len(scan_motors) == data.ndim
            and sorted(m["n"] for m in scan_motors) == sorted(data.shape)
        ):
            axes, labels, units = [], [], []
            remaining = list(scan_motors)
            for size in data.shape:
                match_idx = next(i for i, m in enumerate(remaining) if m["n"] == size)
                motor = remaining.pop(match_idx)
                used_scan.add(motor["name"])
                axes.append(np.asarray(motor["axis"], dtype=float))
                labels.append(motor["name"])
                units.append(motor["unit"] or self._unit_from_label(motor["name"]))
            return data, axes, labels, units

        axes, labels, units = [], [], []
        for index in range(data.ndim):
            size = data.shape[index]
            if index < n_scale:
                offset = float(offsets[index])
                delta = float(deltas[index]) if float(deltas[index]) != 0 else 1.0
                axis = offset + np.arange(size, dtype=float) * delta
                raw_label = _as_str(unit_names[index])
                label, unit = self._normalize_axis_label(raw_label)
                # If Low_Level has a motor with this exact length, prefer its name.
                for motor in scan_motors:
                    if motor["name"] not in used_scan and motor["n"] == size:
                        label = motor["name"]
                        unit = motor["unit"] or unit
                        axis = np.asarray(motor["axis"], dtype=float)
                        used_scan.add(motor["name"])
                        break
                axes.append(axis)
                labels.append(label)
                units.append(unit)
                continue

            motor_axis = self._motor_axis_for_size(size, scan_motors, motors, used_scan)
            if motor_axis is not None:
                axes.append(motor_axis[0])
                labels.append(motor_axis[1])
                units.append(motor_axis[2])
            else:
                axes.append(np.arange(size, dtype=float))
                labels.append(f"Index_{index}")
                units.append("index")

        labels = self._dedupe_labels(labels)
        return data, axes, labels, units

    def _motor_axis_for_size(
        self,
        size: int,
        scan_motors: List[Dict[str, Any]],
        motors: Dict[str, np.ndarray],
        used_scan: set,
    ) -> Optional[Tuple[np.ndarray, str, str]]:
        # 1) Declared scan motor whose actual axis length matches this dim
        #    (0D-resolved axes already reflect aborted / partial scans).
        for motor in scan_motors:
            key = motor.get("raw_name") or motor["name"]
            if key in used_scan or motor["name"] in used_scan:
                continue
            axis = motor.get("axis")
            if axis is not None and len(axis) == size:
                used_scan.add(key)
                used_scan.add(motor["name"])
                return (
                    np.asarray(axis, dtype=float),
                    motor["name"],
                    motor.get("unit") or self._unit_from_label(motor["name"]),
                )
            # Planned linspace longer than written data: truncate.
            if axis is not None and motor.get("planned_n") and len(axis) >= size:
                used_scan.add(key)
                used_scan.add(motor["name"])
                return (
                    np.asarray(axis[:size], dtype=float),
                    motor["name"],
                    motor.get("unit") or self._unit_from_label(motor["name"]),
                )

        # 2) Preferred 0D channels by name, exact length match
        for preferred in self.SCAN_AXIS_PRIORITY:
            if preferred in used_scan:
                continue
            arr = motors.get(preferred)
            if arr is not None and arr.size == size:
                label, unit = self._display_motor(preferred)
                used_scan.add(preferred)
                used_scan.add(label)
                return arr.astype(float), label, unit

        # 3) Any remaining 0D channel: unique-grid first, then raw length.
        #    Skip cryostats / time / counters.
        candidates = []
        for name, arr in motors.items():
            if name in used_scan:
                continue
            lower = name.lower()
            if any(token in lower for token in self.SCAN_AXIS_AVOID):
                continue
            unique = np.unique(np.round(arr, 6))
            if len(unique) == size:
                candidates.append((0, name, unique.astype(float)))
            elif arr.size == size and size > 1:
                candidates.append((1, name, arr.astype(float)))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            _, name, axis = candidates[0]
            label, unit = self._display_motor(name)
            used_scan.add(name)
            used_scan.add(label)
            return axis, label, unit
        return None

    @staticmethod
    def _normalize_axis_label(raw: str) -> Tuple[str, str]:
        text = raw.strip()
        lower = text.lower()
        if lower in {"ev", "energy", "kinetic energy", "binding energy"}:
            return "Energy", "eV"
        if lower in {"deg", "degree", "degrees", "angle"}:
            return "Slit Angle", "deg"
        if lower in {"pixel", "pixels", "px"}:
            return "Detector Pixel", "px"
        if "sample x" in lower:
            return "Sample X", "um" if "um" not in lower else "um"
        if "sample y" in lower:
            return "Sample Y", "um"
        if "sample z" in lower:
            return "Sample Z", "um"
        unit = "eV" if "ev" in lower else "deg" if "deg" in lower else "um" if "um" in lower else "a.u."
        return text, unit

    @staticmethod
    def _unit_from_label(label: str) -> str:
        lower = label.lower()
        if "energy" in lower or lower.endswith("ev"):
            return "eV"
        if "angle" in lower or "theta" in lower or "phi" in lower or "beta" in lower or "defl" in lower:
            return "deg"
        if any(token in lower for token in ("sample", "scan", "stage", "piezo", "optics")):
            return "um"
        if "pixel" in lower:
            return "px"
        if "time" in lower:
            return "s"
        return "a.u."

    @staticmethod
    def _dedupe_labels(labels: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        out = []
        for label in labels:
            count = seen.get(label, 0)
            seen[label] = count + 1
            out.append(label if count == 0 else f"{label}_{count + 1}")
        return out

    # ------------------------------------------------------------------ metadata

    def _build_metadata(
        self,
        *,
        headers: Dict[str, Dict[str, Any]],
        comments: Dict[str, List[str]],
        daq: Dict[str, Any],
        is_fixed: bool,
        source_path: str,
        scan_motors: List[Dict[str, Any]],
        motors: Dict[str, np.ndarray],
        log_row: Optional[Dict[str, str]],
        shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        hv = daq.get("SF_HV") or daq.get("SS_HV")
        pass_energy = daq.get("SFPE_0") or daq.get("SSPE_0")
        lens = daq.get("SFLNM0") or daq.get("SSLNM0")
        slit = daq.get("SF_SLITN") or daq.get("SS_ESLIN") or daq.get("SF_SLIT") or daq.get("SS_ESLIT")
        image_mode = daq.get("SSOD_0") or daq.get("SFOD_0")
        bin_angle = daq.get("SSBA_0") or daq.get("SFBA_0")
        bin_energy = daq.get("SSBE_0") or daq.get("SFBE0") or daq.get("SFBE_0")
        frames = daq.get("SSFR_0") or daq.get("SFFR_0")
        sweeps = daq.get("SSSW0") or daq.get("SFSW0")

        motor_summary = {}
        for name, arr in motors.items():
            if not arr.size:
                continue
            finite = arr[np.isfinite(arr)]
            motor_summary[name] = {
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "n": int(arr.size),
            }

        aborted = any(bool(m.get("aborted")) for m in scan_motors)

        metadata: Dict[str, Any] = {
            "Facility": "MAESTRO",
            "Beamline": "ALS 7.0.2",
            "Source_File": self.filename,
            "Source_Path": self.filepath,
            "Intensity_Path": source_path,
            "Analyzer_Mode": "Fixed" if is_fixed else "Swept",
            "Image_Mode": image_mode,
            "Binning_Angle": bin_angle,
            "Binning_Energy": bin_energy,
            "Frames": frames,
            "Sweeps": sweeps,
            "Photon_Energy_eV": float(hv) if isinstance(hv, (int, float)) else hv,
            "Pass_Energy": pass_energy,
            "Lens_Mode": lens,
            "Entrance_Slit": slit,
            "Scan_Name": headers.get("Low_Level_Scan", {}).get("LWLVNM"),
            "Scan_Motors": [
                {
                    "name": m["name"],
                    "raw_name": m.get("raw_name", m["name"]),
                    "unit": m.get("unit"),
                    "n": m.get("n"),
                    "planned_n": m.get("planned_n"),
                    "aborted": bool(m.get("aborted")),
                }
                for m in scan_motors
            ],
            "Aborted_Scan": aborted,
            "Comments": comments,
            "Headers": headers,
            "Motor_Summary": motor_summary,
            "Shape": list(shape),
            # Full 0D traces for DataTreeBuilder → /raw/motors
            "motors": {name: arr for name, arr in motors.items()},
        }

        if log_row:
            metadata["Measurement_Log"] = log_row
            metadata["Measurement_Type"] = log_row.get("Measurement Type")
            metadata["Log_Comments"] = log_row.get("Pre/Post Comments")
            metadata["Log_Photon_Energy"] = log_row.get("Photon Energy (eV)")
            metadata["Log_Polarization"] = log_row.get("Polarization")
            metadata["Log_Manipulator"] = log_row.get("Manipulator Position")
            metadata["Log_Deflection"] = log_row.get("Deflection Settings")
            metadata["Log_Analyzer"] = log_row.get("Analyzer Settings")
            metadata["Log_Energy_Resolution"] = log_row.get("Energy Resolution")

        return metadata
