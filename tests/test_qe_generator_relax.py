import os
from pymatgen.core import Lattice, Structure
from tensorspec.core.dft.qe_generator import QEInputGenerator


def test_write_relax_ions_fixed_cell(tmp_path, monkeypatch):
    lat = Lattice.hexagonal(2.5, 23.4)
    s = Structure(lat, ["C", "C"], [[1 / 3, 2 / 3, 0.4], [2 / 3, 1 / 3, 0.4]])
    gen = QEInputGenerator(s)
    monkeypatch.setattr(
        gen,
        "_generate_atomic_species",
        lambda out_dir, use_soc=False: " C  12.01  C.upf",
    )
    path = gen.write_relax_input(
        str(tmp_path), ecutwfc=40, kmesh=(4, 4, 1), vdw_dft_d3=True
    )
    text = open(path).read()
    assert "calculation = 'relax'" in text
    assert "vdw_corr = 'dft-d3'" in text
    assert "CELL_PARAMETERS" in text
    assert "ATOMIC_POSITIONS" in text
    assert "&IONS" in text
    assert "forc_conv_thr = 1.0d-3" in text
    assert "&CELL" not in text


def test_write_vc_relax_flag(tmp_path, monkeypatch):
    lat = Lattice.hexagonal(2.5, 23.4)
    s = Structure(lat, ["C", "C"], [[1 / 3, 2 / 3, 0.4], [2 / 3, 1 / 3, 0.4]])
    gen = QEInputGenerator(s)
    monkeypatch.setattr(
        gen,
        "_generate_atomic_species",
        lambda out_dir, use_soc=False: " C  12.01  C.upf",
    )
    path = gen.write_relax_input(
        str(tmp_path),
        ecutwfc=40,
        kmesh=(4, 4, 1),
        calculation="vc-relax",
        vdw_dft_d3=False,
    )
    text = open(path).read()
    assert "calculation = 'vc-relax'" in text
    assert "vdw_corr" not in text
    assert "&CELL" in text


def test_relax_if_pos_appended(tmp_path, monkeypatch):
    lat = Lattice.hexagonal(2.5, 23.4)
    s = Structure(
        lat,
        ["C", "C", "B", "N"],
        [[1 / 3, 2 / 3, 0.4], [2 / 3, 1 / 3, 0.4], [1 / 3, 2 / 3, 0.6], [2 / 3, 1 / 3, 0.6]],
    )
    gen = QEInputGenerator(s)
    monkeypatch.setattr(
        gen,
        "_generate_atomic_species",
        lambda out_dir, use_soc=False: " C  12.01  C.upf\n B  10.81  B.upf\n N  14.01  N.upf",
    )
    if_pos = [(0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1)]
    path = gen.write_relax_input(str(tmp_path), ecutwfc=40, kmesh=(4, 4, 1), if_pos=if_pos)
    text = open(path).read()
    assert " 0  0  0" in text
    assert " 1  1  1" in text


def test_scf_nscf_vdw_flag(tmp_path, monkeypatch):
    lat = Lattice.hexagonal(2.5, 23.4)
    s = Structure(lat, ["C", "C"], [[1 / 3, 2 / 3, 0.4], [2 / 3, 1 / 3, 0.4]])
    gen = QEInputGenerator(s)
    monkeypatch.setattr(
        gen,
        "_generate_atomic_species",
        lambda out_dir, use_soc=False: " C  12.01  C.upf",
    )
    scf_path = gen.write_scf_input(str(tmp_path), vdw_dft_d3=True)
    nscf_path = gen.write_nscf_input(str(tmp_path), vdw_dft_d3=True)
    assert "vdw_corr = 'dft-d3'" in open(scf_path).read()
    assert "vdw_corr = 'dft-d3'" in open(nscf_path).read()


def _hex_carbon_gen(monkeypatch):
    lat = Lattice.hexagonal(2.5, 23.4)
    s = Structure(lat, ["C", "C"], [[1 / 3, 2 / 3, 0.4], [2 / 3, 1 / 3, 0.4]])
    gen = QEInputGenerator(s)
    monkeypatch.setattr(
        gen,
        "_generate_atomic_species",
        lambda out_dir, use_soc=False: " C  12.01  C.upf",
    )
    return gen, s


def test_rewrite_scf_uses_new_coords(tmp_path, monkeypatch):
    gen, s = _hex_carbon_gen(monkeypatch)
    gen.write_scf_input(str(tmp_path), ecutwfc=40, kmesh=(4, 4, 1))
    assert "0.400000" in open(tmp_path / "scf.in").read()
    assert "0.450000" not in open(tmp_path / "scf.in").read()

    s2 = s.copy()
    s2.translate_sites(list(range(len(s2))), [0, 0, 0.05], frac_coords=True)
    gen.apply_structure(s2)
    gen.write_scf_input(str(tmp_path), ecutwfc=40, kmesh=(4, 4, 1))

    text = open(tmp_path / "scf.in").read()
    assert "0.450000" in text
    assert "0.400000" not in text


def test_rewrite_nscf_uses_new_coords(tmp_path, monkeypatch):
    gen, s = _hex_carbon_gen(monkeypatch)
    gen.write_nscf_input(str(tmp_path), ecutwfc=40, kmesh=(4, 4, 1))
    assert "0.400000" in open(tmp_path / "nscf.in").read()

    s2 = s.copy()
    s2.translate_sites(list(range(len(s2))), [0, 0, 0.05], frac_coords=True)
    gen.apply_structure(s2)
    gen.write_nscf_input(str(tmp_path), ecutwfc=40, kmesh=(4, 4, 1))

    text = open(tmp_path / "nscf.in").read()
    assert "0.450000" in text
    assert "0.400000" not in text


def test_rewrite_wannier_uses_new_coords(tmp_path, monkeypatch):
    gen, s = _hex_carbon_gen(monkeypatch)
    gen.write_wannier90_input(str(tmp_path), kmesh=(4, 4, 1))
    assert "0.400000" in open(tmp_path / "wannier90.win").read()

    s2 = s.copy()
    s2.translate_sites(list(range(len(s2))), [0, 0, 0.05], frac_coords=True)
    gen.apply_structure(s2)
    gen.write_wannier90_input(str(tmp_path), kmesh=(4, 4, 1))

    text = open(tmp_path / "wannier90.win").read()
    assert "0.450000" in text
    assert "0.400000" not in text
