"""Detect QE Fermi energy beside a Wannier/QE work directory."""

from __future__ import annotations

import os
from typing import Optional, Tuple


def _parse_fermi_line(line: str) -> Optional[float]:
    if "the Fermi energy is" in line:
        try:
            return float(line.split("is")[1].split("ev")[0].strip())
        except (IndexError, ValueError):
            return None
    return None


def _parse_fermi_energy_txt(path: str) -> Optional[float]:
    try:
        with open(path, "r") as f:
            for line in f:
                parsed = _parse_fermi_line(line)
                if parsed is not None:
                    return parsed
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    return float(stripped.split()[0])
                except ValueError:
                    continue
    except OSError:
        return None
    return None


def _parse_qe_out_fermi(path: str) -> Optional[float]:
    try:
        with open(path, "r") as f:
            for line in f:
                parsed = _parse_fermi_line(line)
                if parsed is not None:
                    return parsed
    except OSError:
        return None
    return None


def detect_qe_fermi_eV(work_dir: str) -> Tuple[float, str]:
    """
    Auto-detect QE Fermi (eV) from a Wannier/QE folder.

    Order: nscf.out → scf.out → FERMI_ENERGY.txt → (0.0, \"none\").
    """
    if not work_dir:
        return 0.0, "none"

    nscf = os.path.join(work_dir, "nscf.out")
    if os.path.isfile(nscf):
        val = _parse_qe_out_fermi(nscf)
        if val is not None:
            return float(val), "nscf.out"

    scf = os.path.join(work_dir, "scf.out")
    if os.path.isfile(scf):
        val = _parse_qe_out_fermi(scf)
        if val is not None:
            return float(val), "scf.out"

    fermi_txt = os.path.join(work_dir, "FERMI_ENERGY.txt")
    if os.path.isfile(fermi_txt):
        val = _parse_fermi_energy_txt(fermi_txt)
        if val is not None:
            return float(val), "FERMI_ENERGY.txt"

    return 0.0, "none"
