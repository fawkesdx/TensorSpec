"""QE / DFT slab helpers: presets, Miller cleave, and 2D-cell detection."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from pymatgen.core import Structure

from tensorspec.core.crystallography import CrystalEngine

# Educational presets: face × thickness. Custom uses caller hkl / layers / vacuum.
SLAB_PRESETS: Dict[str, Dict[str, Any]] = {
    "thin_001": {"hkl": (0, 0, 1), "num_layers": 1, "vacuum": 15.0},
    "medium_001": {"hkl": (0, 0, 1), "num_layers": 3, "vacuum": 15.0},
    "thick_001": {"hkl": (0, 0, 1), "num_layers": 5, "vacuum": 20.0},
    "thin_111": {"hkl": (1, 1, 1), "num_layers": 1, "vacuum": 15.0},
    "medium_111": {"hkl": (1, 1, 1), "num_layers": 3, "vacuum": 15.0},
    "thick_111": {"hkl": (1, 1, 1), "num_layers": 5, "vacuum": 20.0},
    "thin_110": {"hkl": (1, 1, 0), "num_layers": 1, "vacuum": 15.0},
    "medium_110": {"hkl": (1, 1, 0), "num_layers": 3, "vacuum": 15.0},
}


def list_slab_presets() -> list[dict]:
    """UI-facing preset catalog."""
    out = [
        {
            "id": "custom",
            "label": "Custom (use hkl / layers / vacuum below)",
            "hkl": None,
            "num_layers": None,
            "vacuum": None,
        }
    ]
    for key, meta in SLAB_PRESETS.items():
        h, k, l = meta["hkl"]
        out.append(
            {
                "id": key,
                "label": f"{key.replace('_', ' ')} — ({h}{k}{l}), {meta['num_layers']} layer(s), {meta['vacuum']:.0f} Å vac",
                "hkl": list(meta["hkl"]),
                "num_layers": meta["num_layers"],
                "vacuum": meta["vacuum"],
            }
        )
    return out


def resolve_slab_params(
    preset: str = "custom",
    *,
    h: int = 0,
    k: int = 0,
    l: int = 1,
    num_layers: int = 1,
    vacuum: float = 15.0,
) -> Tuple[Tuple[int, int, int], int, float]:
    """Return (hkl, num_layers, vacuum) after applying a named preset if any."""
    key = (preset or "custom").strip().lower()
    if key in SLAB_PRESETS:
        meta = SLAB_PRESETS[key]
        return meta["hkl"], int(meta["num_layers"]), float(meta["vacuum"])
    return (int(h), int(k), int(l)), max(1, int(num_layers)), float(vacuum)


def suggest_slab_qe(structure: Structure | None) -> bool:
    """True when the cell looks like a slab / 2D stack (large c)."""
    if structure is None:
        return False
    try:
        return float(structure.lattice.c) > 12.0
    except Exception:
        return False


def prepare_slab(
    bulk: Structure,
    *,
    preset: str = "custom",
    h: int = 0,
    k: int = 0,
    l: int = 1,
    num_layers: int = 1,
    vacuum: float = 15.0,
    bond_threshold: float = 3.2,
) -> dict:
    """Cleave bulk into a vacuum slab for QE surface / 2D calculations."""
    hkl, layers, vac = resolve_slab_params(
        preset, h=h, k=k, l=l, num_layers=num_layers, vacuum=vacuum
    )
    slab = CrystalEngine.extract_monolayer_miller(
        bulk,
        hkl=hkl,
        num_layers=layers,
        vacuum=vac,
        bond_threshold=bond_threshold,
    )
    return {
        "structure": slab,
        "hkl": list(hkl),
        "num_layers": layers,
        "vacuum": vac,
        "preset": (preset or "custom").strip().lower(),
        "n_sites": len(slab),
        "formula": slab.composition.reduced_formula,
        "suggest_slab_qe": True,
    }
