"""Optional ML interatomic-potential relaxation (CHGNet / M3GNet via matgl).

Models are loaded lazily. If ``matgl`` / ``ase`` / ``torch`` are not installed,
``list_models`` still describes them and ``relax_structure`` raises a clear error.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Canonical user-facing choices → ordered load attempts (HF id or short matgl name)
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "chgnet": {
        "label": "CHGNet (MatPES-PBE, universal)",
        "description": (
            "Crystal Hamiltonian Graph Network — pretrained universal potential; "
            "good default for stacked 2D heterostructures (fast vs DFT)."
        ),
        "candidates": [
            "CHGNet-PES-MatPES-PBE-2025.2.10",
            "materialyze/CHGNet-PES-MatPES-PBE-2025.2.10",
            "CHGNet-MatPES-PBE-2025.2",
        ],
    },
    "m3gnet": {
        "label": "M3GNet (Materials Project universal)",
        "description": (
            "Materials 3-body Graph Network — Materials Project foundation potential; "
            "broad chemistry coverage for ionic relaxation."
        ),
        "candidates": [
            "M3GNet-MatPES-PBE-2025.1",
            "M3GNet-MP-2021.2.8-PES",
            "materialyze/M3GNet-MatPES-PBE-2025.1",
        ],
    },
}

_POTENTIAL_CACHE: Dict[str, Any] = {}


def mlip_available() -> bool:
    try:
        import matgl  # noqa: F401
        import ase  # noqa: F401
        return True
    except Exception:
        return False


def list_models() -> List[Dict[str, Any]]:
    ready = mlip_available()
    return [
        {
            "id": key,
            "label": meta["label"],
            "description": meta["description"],
            "available": ready,
        }
        for key, meta in MODEL_CATALOG.items()
    ]


def _load_potential(model_id: str):
    if model_id in _POTENTIAL_CACHE:
        return _POTENTIAL_CACHE[model_id]
    if model_id not in MODEL_CATALOG:
        raise ValueError(f"Unknown MLIP model '{model_id}'. Choose: {', '.join(MODEL_CATALOG)}.")
    try:
        import matgl
    except Exception as exc:
        raise RuntimeError(
            "MLIP relaxation needs matgl (+ ase, torch). "
            "Install with: pip install 'matgl' ase torch"
        ) from exc

    errors = []
    for name in MODEL_CATALOG[model_id]["candidates"]:
        try:
            pot = matgl.load_model(name)
            _POTENTIAL_CACHE[model_id] = pot
            return pot
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        f"Could not load any pretrained weights for '{model_id}'. Tried:\n"
        + "\n".join(errors)
    )


def relax_structure(
    structure,
    *,
    model: str = "chgnet",
    fmax: float = 0.1,
    steps: int = 200,
    relax_cell: bool = False,
) -> Dict[str, Any]:
    """
    Relax a pymatgen Structure with a pretrained universal MLIP.

    Returns the relaxed structure plus ASE trajectory summary scalars.
    """
    from pymatgen.core import Structure

    if not isinstance(structure, Structure):
        raise TypeError("structure must be a pymatgen Structure.")
    if len(structure) == 0:
        raise ValueError("Empty structure.")
    if steps < 1 or steps > 2000:
        raise ValueError("steps must be between 1 and 2000.")
    if fmax <= 0 or fmax > 5:
        raise ValueError("fmax must be in (0, 5] eV/Å.")

    pot = _load_potential(model.lower().strip())
    try:
        from matgl.ext.ase import Relaxer
    except Exception as exc:
        raise RuntimeError(f"matgl ASE Relaxer unavailable: {exc}") from exc

    relaxer = Relaxer(potential=pot, relax_cell=bool(relax_cell))
    result = relaxer.relax(structure, fmax=float(fmax), steps=int(steps))

    # matgl returns dict with "final_structure" and trajectory; be defensive
    if isinstance(result, dict):
        final = result.get("final_structure") or result.get("structure")
        traj = result.get("trajectory")
    else:
        final = result
        traj = None
    if final is None:
        raise RuntimeError("MLIP relaxer returned no final structure.")

    # Preserve layer tags when present
    if "layer_tag" in structure.site_properties and "layer_tag" not in final.site_properties:
        if len(final) == len(structure):
            final.add_site_property("layer_tag", structure.site_properties["layer_tag"])

    n_steps = None
    final_energy = None
    if traj is not None:
        try:
            n_steps = int(len(traj))
        except Exception:
            n_steps = None
        try:
            energies = getattr(traj, "energies", None)
            if energies is not None and len(energies):
                final_energy = float(energies[-1])
        except Exception:
            pass

    return {
        "structure": final,
        "model": model.lower().strip(),
        "fmax": float(fmax),
        "steps_requested": int(steps),
        "steps_taken": n_steps,
        "relax_cell": bool(relax_cell),
        "final_energy_eV": final_energy,
        "n_sites": len(final),
        "formula": final.composition.reduced_formula,
    }
