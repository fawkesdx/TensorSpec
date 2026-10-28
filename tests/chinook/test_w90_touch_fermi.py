"""W90 Dirac/semimetal midgap preferred over an offset QE EF for plots.

Synthetic drives only: a coarse-mesh graphene run puts the K-point touch
~0.4 eV below the QE Fermi level (midgap must win), while a converged run
puts it within smearing of EF (QE EF must win).  Numbers mirror the
graphene_dft_drive 4x4 and 12x12 runs but nothing here reads scratch/.
"""
from pathlib import Path

import pytest

from tensorspec.core.dft.qe_fermi import (
    detect_plot_fermi_eV,
    detect_qe_fermi_eV,
    detect_w90_touch_fermi_eV,
)

QE_EF = -2.3512
# Deep accidental degeneracy at Γ (outside the 2 eV window around QE EF).
GAMMA_ENERGIES = [-21.85, -9.968, -5.416, -5.416, 0.886, 1.614, 1.802, 2.772]
M_ENERGIES = [-16.624, -15.584, -8.787, -4.725, -0.691, 4.998, 8.736, 8.885]
# Rows (1-based) of the high-symmetry points along the synthetic path.
LABEL_ROWS = [("G", 1), ("M", 3), ("K", 5)]
N_PTS = 6


def _write_drive(tmp_path: Path, k_energies, qe_ef=QE_EF, with_qe=True) -> str:
    """Write wannier90_band.dat / labelinfo (and nscf.out) in W90 layout."""
    per_row = {1: GAMMA_ENERGIES, 3: M_ENERGIES, 5: k_energies}
    rows = [per_row.get(r, M_ENERGIES) for r in range(1, N_PTS + 1)]
    n_bands = len(k_energies)
    lines = []
    for b in range(n_bands):
        for r, energies in enumerate(rows):
            lines.append(f"  {0.1 * r:.8E} {sorted(energies)[b]:.8E}")
        lines.append("")  # blank line separates bands, as W90 writes it
    (tmp_path / "wannier90_band.dat").write_text("\n".join(lines) + "\n")
    (tmp_path / "wannier90_band.labelinfo.dat").write_text(
        "".join(f"{lab} {row} 0.0 0.0 0.0 0.0\n" for lab, row in LABEL_ROWS)
    )
    if with_qe:
        (tmp_path / "nscf.out").write_text(
            f"     the Fermi energy is   {qe_ef:.4f} ev\n"
        )
    return str(tmp_path)


def test_offset_touch_prefers_midgap(tmp_path):
    # Coarse 4x4-mesh artefact: 29 meV false gap at K, 0.44 eV below QE EF.
    wd = _write_drive(
        tmp_path, [-14.751, -14.024, -13.071, -2.806, -2.777, 7.443, 8.544, 9.159]
    )
    qe, qe_src = detect_qe_fermi_eV(wd)
    assert qe_src == "nscf.out" and qe == pytest.approx(QE_EF)

    touch, touch_src = detect_w90_touch_fermi_eV(wd)
    assert touch == pytest.approx(-2.7915, abs=1e-3)
    assert touch_src == "W90 midgap@K"
    # Must not pick the deep Γ accidental degeneracy (~−5.4 eV).
    assert abs(touch - qe) < 1.0

    plot_ef, plot_src = detect_plot_fermi_eV(wd)
    assert plot_ef == pytest.approx(touch)
    assert "midgap" in plot_src and "nscf.out" in plot_src


def test_aligned_touch_falls_back_to_qe_ef(tmp_path):
    # Converged 12x12 run: true degeneracy at K within smearing of QE EF.
    wd = _write_drive(
        tmp_path, [-14.948, -14.948, -12.909, -2.3702, -2.3701, 8.254, 10.495, 10.496]
    )
    touch, touch_src = detect_w90_touch_fermi_eV(wd)
    assert touch == pytest.approx(-2.37015, abs=1e-3)
    assert touch_src == "W90 midgap@K"

    plot_ef, plot_src = detect_plot_fermi_eV(wd)
    assert plot_ef == pytest.approx(QE_EF)
    assert plot_src == "nscf.out"
    assert "midgap" not in plot_src


def test_no_qe_fermi_keeps_smallest_gap_anywhere(tmp_path):
    # Without a QE EF there is no window, so the Γ degeneracy (gap 0) wins.
    wd = _write_drive(
        tmp_path,
        [-14.751, -14.024, -13.071, -2.806, -2.777, 7.443, 8.544, 9.159],
        with_qe=False,
    )
    assert detect_qe_fermi_eV(wd) == (0.0, "none")
    touch, touch_src = detect_w90_touch_fermi_eV(wd)
    assert touch == pytest.approx(-5.416)
    assert touch_src == "W90 midgap@G"


def test_missing_band_files_returns_none(tmp_path):
    (tmp_path / "nscf.out").write_text("     the Fermi energy is   -2.3512 ev\n")
    assert detect_w90_touch_fermi_eV(str(tmp_path)) == (None, "none")
    assert detect_plot_fermi_eV(str(tmp_path)) == (pytest.approx(QE_EF), "nscf.out")
