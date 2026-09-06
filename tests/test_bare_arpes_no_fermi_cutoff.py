"""Bare-band ARPES intensity: no Fermi–Dirac cutoff."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import _bare_intensity


def test_bare_intensity_keeps_weight_above_fermi():
    # One k-point, one band at +0.5 eV (unoccupied if EF=0).
    energy_axis = np.linspace(-1.0, 1.0, 41)
    exp = SimpleNamespace(val=np.array([[0.5]], dtype=float))
    ctx = {
        "exp": exp,
        "energy_axis": energy_axis,
        "num_x": 1,
        "num_y": 1,
        "num_e": energy_axis.size,
        "se_width": 0.05,
        "T": 10.0,
    }
    Ig = _bare_intensity(ctx)
    assert Ig.shape == (1, 1, energy_axis.size)
    i_above = int(np.argmin(np.abs(energy_axis - 0.5)))
    i_below = int(np.argmin(np.abs(energy_axis - (-0.5))))
    # Peak near the band energy above EF must remain strong (no FD wipeout).
    assert Ig[0, 0, i_above] > 0.1 * Ig[0, 0].max()
    assert Ig[0, 0, i_above] > Ig[0, 0, i_below]
