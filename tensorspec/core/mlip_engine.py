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
    models = [
        {
            "id": key,
            "label": meta["label"],
            "description": meta["description"],
            "available": ready,
            "kind": "relax",
        }
        for key, meta in MODEL_CATALOG.items()
    ]
    models.append(
        {
            "id": "megnet_gap",
            "label": "MEGNet band gap (multi-fidelity)",
            "description": (
                "Predicts a scalar band gap (not full E(k)). Useful when the stack is too "
                "large for DFT bands — gives a quick electronic scale (PBE/HSE/… fidelity)."
            ),
            "available": ready,
            "kind": "gap",
        }
    )
    return models


def predict_band_gap(
    structure,
    *,
    fidelity: str = "PBE",
) -> Dict[str, Any]:
    """
    Predict band gap with pretrained MEGNet multi-fidelity model.

    This is a scalar surrogate — not a full band-structure dispersion.
    """
    from pymatgen.core import Structure

    if not isinstance(structure, Structure):
        raise TypeError("structure must be a pymatgen Structure.")
    fidelity_map = {"PBE": 0, "GLLB-SC": 1, "HSE": 2, "SCAN": 3}
    key = fidelity.upper().replace("_", "-")
    # normalize
    aliases = {
        "PBE": "PBE",
        "GLLB": "GLLB-SC",
        "GLLB-SC": "GLLB-SC",
        "HSE": "HSE",
        "SCAN": "SCAN",
    }
    fid_name = aliases.get(key, aliases.get(fidelity.upper(), "PBE"))
    fid_idx = fidelity_map[fid_name]

    try:
        import matgl
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Band-gap prediction needs matgl (+ torch). "
            "Install with: pip install matgl torch"
        ) from exc

    candidates = [
        "MEGNet-MP-2019.4.1-BandGap-mfi",
    ]
    # Also try HF-style ids if short name fails
    errors = []
    model = None
    loaded_as = None
    for name in candidates:
        try:
            model = matgl.load_model(name)
            loaded_as = name
            break
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if model is None:
        raise RuntimeError(
            "Could not load MEGNet band-gap weights. Tried:\n" + "\n".join(errors)
        )

    state = torch.tensor([fid_idx])
    try:
        gap = model.predict_structure(structure=structure, state_attr=state)
    except TypeError:
        # Older API variants
        gap = model.predict_structure(structure)
    gap_eV = float(gap)
    return {
        "gap_eV": gap_eV,
        "fidelity": fid_name,
        "fidelity_index": fid_idx,
        "model": loaded_as,
        "formula": structure.composition.reduced_formula,
        "n_sites": len(structure),
        "note": (
            "Scalar band-gap estimate only — not a full E(k) dispersion. "
            "For stack/twist intuition use DFT Suite TB bands (folded supercell path) "
            "or the primitive-hex reference path."
        ),
    }


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
