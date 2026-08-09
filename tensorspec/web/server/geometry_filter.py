"""Filter crystal geometry and pymatgen Structure by omitted atom indices."""

from __future__ import annotations

import copy

from pymatgen.core import Structure


def normalize_omit_indices(omit: list[int] | None, n_atoms: int) -> set[int]:
    """Non-negative ints in [0, n_atoms); drop OOR/dupes."""
    if not omit:
        return set()
    return {i for i in omit if isinstance(i, int) and 0 <= i < n_atoms}


def filter_geometry_atoms_bonds(
    atoms: list,
    bonds: list,
    omit: set[int],
) -> tuple[list, list]:
    """Keep atoms not in omit; bonds whose both ends survive; remap bond i,j to new compact indices."""
    kept_atoms: list = []
    old_to_new: dict[int, int] = {}
    for old_idx, atom in enumerate(atoms):
        if old_idx in omit:
            continue
        old_to_new[old_idx] = len(kept_atoms)
        kept_atoms.append(atom)

    kept_bonds: list = []
    for bond in bonds:
        if bond.i in omit or bond.j in omit:
            continue
        if bond.i not in old_to_new or bond.j not in old_to_new:
            continue
        remapped = copy.copy(bond)
        remapped.i = old_to_new[bond.i]
        remapped.j = old_to_new[bond.j]
        kept_bonds.append(remapped)

    return kept_atoms, kept_bonds


def filter_structure_by_omit(structure: Structure, omit: set[int]) -> Structure:
    """Return new Structure without sites at omit indices (OOR ignored)."""
    valid_omit = {i for i in omit if 0 <= i < len(structure)}
    if not valid_omit:
        return structure.copy()
    out = structure.copy()
    for idx in sorted(valid_omit, reverse=True):
        out.remove_sites([idx])
    return out
