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
