from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure

from tensorspec.core.crystallography import CrystalEngine


def _mono(a: float, species=("C", "C")) -> Structure:
    lat = Lattice.hexagonal(a, 25.0)
    return Structure(lat, list(species), [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])


def _layer(struct, z=0.0, twist=0.0, sc_x=1, sc_y=1):
    return {
        "struct": struct,
        "sc_x": sc_x,
        "sc_y": sc_y,
        "z_shift": z,
        "twist": twist,
    }


def test_classify_empty():
    assert CrystalEngine.classify_stack_for_dft([]) == "empty"


def test_classify_aligned_n_layers():
    g = _mono(2.46)
    layers = [_layer(g, 0.0), _layer(g, 3.4), _layer(g, 6.8)]
    assert CrystalEngine.classify_stack_for_dft(layers) == "aligned"


def test_classify_twist_two_layers():
    layers = [_layer(_mono(2.46), 0.0, 0.0), _layer(_mono(2.50), 3.4, 30.0)]
    assert CrystalEngine.classify_stack_for_dft(layers) == "twist"


def test_classify_reject_three_with_twist():
    g = _mono(2.46)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, 10.0), _layer(g, 6.8, 0.0)]
    assert CrystalEngine.classify_stack_for_dft(layers) == "reject_multitwist"


def test_suggest_ref_prefers_closer_lattice():
    # identical twins -> either ok; mismatched pair -> closer match wins
    layers = [_layer(_mono(2.46), 0.0), _layer(_mono(2.50), 3.4)]
    idx, strains = CrystalEngine.suggest_reference_layer(layers)
    assert len(strains) == 2
    assert idx in (0, 1)
    assert strains[idx] == min(strains)


def test_aligned_bilayer_not_dummy_and_vacuum():
    g = _mono(2.46)
    layers = [_layer(g, 0.0), _layer(g, 3.4)]
    vacuum = 20.0
    s = CrystalEngine.build_dft_aligned_stack(layers, vacuum_ang=vacuum, ref_idx=0)
    assert s.lattice.a < 100.0
    assert s.lattice.a != 500.0
    thickness = 3.4  # |z1-z0| from spinboxes (same placement)
    # c must be roughly thickness + vacuum (allow builder offset conventions ±2 Å)
    assert abs(s.lattice.c - (thickness + vacuum)) < 2.5
    assert "layer_tag" in s.site_properties
    assert len(s) == 4


def test_aligned_mismatch_flag():
    layers = [_layer(_mono(2.46), 0.0), _layer(_mono(2.50), 3.4)]
    assert CrystalEngine.aligned_needs_strain_dialog(layers, ref_idx=0) is True
    same = [_layer(_mono(2.46), 0.0), _layer(_mono(2.46), 3.4)]
    assert CrystalEngine.aligned_needs_strain_dialog(same, ref_idx=0) is False


def test_twist_two_layer_cell_not_dummy():
    layers = [_layer(_mono(2.46), 0.0, 0.0), _layer(_mono(2.50, ("B", "N")), 3.4, 30.0)]
    suggested, strains = CrystalEngine.suggest_reference_layer(layers)
    struct, info = CrystalEngine.build_dft_twist_stack(layers, vacuum_ang=20.0, ref_idx=suggested)
    assert struct.lattice.a < 499.0
    assert struct.lattice.c > 20.0
    assert info["status"] in ("commensurate", "incommensurate", "perfect_alignment")
    assert "layer_tag" in struct.site_properties


def test_twist_requires_two_layers():
    layers = [_layer(_mono(2.46), 0.0, 5.0)]
    try:
        CrystalEngine.build_dft_twist_stack(layers, 20.0, 0)
        assert False, "expected ValueError"
    except ValueError:
        pass

def test_twist_commensurate_fills_moire_cell():
    """Identical lattices + twist marked commensurate must fill moiré area, not one SC."""
    g = _mono(2.46)
    twist = 21.5  # calculate_moire_superlattice marks this commensurate (n_cells≈3)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, twist)]
    moire = CrystalEngine.calculate_moire_superlattice(g, g, 0.0, twist)
    assert moire["status"] == "commensurate"
    n_cells = int(moire["n_cells"])
    assert n_cells >= 2

    struct, info = CrystalEngine.build_dft_twist_stack(layers, vacuum_ang=20.0, ref_idx=0)
    assert info["status"] == "commensurate"

    ab = np.asarray(moire["matrix"], dtype=float)
    area_m = abs(np.linalg.det(ab))
    area_l = abs(np.linalg.det(CrystalEngine._inplane_2x2(g)))
    # 2 atoms/cell × 2 layers × area ratio
    expected = 2 * 2 * (area_m / area_l)
    # Must be far above a single primitive bilayer (4); allow boundary/dedup slack
    assert len(struct) > 4 * n_cells
    assert abs(len(struct) - expected) / expected < 0.35
    # Identical lattices → same fill per layer (no half-open/dedup imbalance)
    tags = struct.site_properties["layer_tag"]
    n_l1 = sum(1 for t in tags if t.endswith("_L1"))
    n_l2 = sum(1 for t in tags if t.endswith("_L2"))
    assert n_l1 == n_l2, f"identical-lattice layers unequal: L1={n_l1} L2={n_l2}"
    assert n_l1 * 2 == len(struct)


def test_twist_commensurate_empty_tile_raises(monkeypatch):
    """Keep hard fail if tiling yields no atoms."""
    g = _mono(2.46)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, 21.5)]

    def _empty(*args, **kwargs):
        return [], [], []

    monkeypatch.setattr(CrystalEngine, "_tile_into_cell", staticmethod(_empty))
    try:
        CrystalEngine.build_dft_twist_stack(layers, 20.0, 0)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "no atoms" in str(e).lower() or "tiling" in str(e).lower()

