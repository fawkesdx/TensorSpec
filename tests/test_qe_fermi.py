"""Tests for QE Fermi detection beside Wannier packs."""

from __future__ import annotations

from tensorspec.core.dft.qe_fermi import detect_qe_fermi_eV


def test_detect_prefers_nscf_over_scf_and_txt(tmp_path):
    (tmp_path / "nscf.out").write_text("     the Fermi energy is    11.1000 ev\n")
    (tmp_path / "scf.out").write_text("     the Fermi energy is    10.0000 ev\n")
    (tmp_path / "FERMI_ENERGY.txt").write_text("     the Fermi energy is    12.4202 ev\n")
    val, src = detect_qe_fermi_eV(str(tmp_path))
    assert src == "nscf.out"
    assert abs(val - 11.1) < 1e-9


def test_detect_scf_when_no_nscf(tmp_path):
    (tmp_path / "scf.out").write_text("     the Fermi energy is    9.8765 ev\n")
    val, src = detect_qe_fermi_eV(str(tmp_path))
    assert src == "scf.out"
    assert abs(val - 9.8765) < 1e-9


def test_detect_fermi_energy_txt(tmp_path):
    (tmp_path / "FERMI_ENERGY.txt").write_text(
        "     the Fermi energy is    12.4202 ev\n"
    )
    val, src = detect_qe_fermi_eV(str(tmp_path))
    assert src == "FERMI_ENERGY.txt"
    assert abs(val - 12.4202) < 1e-9


def test_detect_fermi_energy_txt_bare_float(tmp_path):
    (tmp_path / "FERMI_ENERGY.txt").write_text("12.4202\n")
    val, src = detect_qe_fermi_eV(str(tmp_path))
    assert src == "FERMI_ENERGY.txt"
    assert abs(val - 12.4202) < 1e-9


def test_detect_none(tmp_path):
    val, src = detect_qe_fermi_eV(str(tmp_path))
    assert val == 0.0
    assert src == "none"
