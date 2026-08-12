"""Suggest QE/Wannier nbnd from crystal sites (old DFT Suite rule)."""

from __future__ import annotations


def suggest_nbnd_base(structure) -> int:
    """Per site: transition metal or Z>30 → 9 (s+p+d); else 4 (s+p).

    Matches the removed Qt ``load_workspace_structure`` Wannier band count.
    SOC doubling is applied in the UI, not here.
    """
    total = 0
    for site in structure:
        el = site.specie
        if getattr(el, "is_transition_metal", False) or int(el.number) > 30:
            total += 9
        else:
            total += 4
    return max(1, int(total))
