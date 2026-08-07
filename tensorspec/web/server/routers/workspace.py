"""Workspace endpoints: list, inspect, and seed the caller's session data.

This module routes requests to `core/`; it performs no physics of its own.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from tensorspec.core.crystallography import CrystalEngine
from tensorspec.core.data_models import TensorData
from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.schemas import (
    DemoSeedRequest,
    ItemDetail,
    WorkspaceItem,
    WorkspaceListing,
)
from tensorspec.web.server.session import Session, current_session

router = APIRouter(prefix="/api/workspace", tags=["workspace"])

VIEWABLE_TYPES = {"spectroscopy_tree", "crystal_structure"}


def _pull_raw(workspace: WorkspaceManager, name: str):
    return workspace.pull_tensor_data(name)


def _describe(name: str, item: dict, workspace: WorkspaceManager) -> tuple[str, str]:
    """Return the (type, dims) pair the variable tree displays for an item."""
    item_type = item.get("type", "Unknown")

    if item_type == "crystal_structure":
        structure = item.get("structure")
        if structure is not None:
            return "Crystal Structure", f"{structure.composition.reduced_formula}, {len(structure)} sites"
        return "Crystal Structure", "3D Basis Vectors"

    if item_type == "band_structure":
        eigenvalues = item.get("eigenvalues")
        shape = getattr(eigenvalues, "shape", None)
        return "Band Structure", f"E(k) Dispersion {shape}" if shape else "E(k) Dispersion Data"

    if item_type == "spectroscopy_tree":
        tensor = _pull_raw(workspace, name)
        if tensor is not None:
            labels = ", ".join(f"'{label}'" for label in tensor.labels)
            return "Spectroscopy DataTree", f"[{labels}] -> {tensor.value.shape}"
        return "Spectroscopy DataTree", "N-Dimensional Tensor"

    return item_type, "N/A"


@router.get("/items", response_model=WorkspaceListing)
def list_items(session: Session = Depends(current_session)) -> WorkspaceListing:
    """Every variable currently held in this session's workspace."""
    workspace = session.workspace
    items = []
    for name, item in workspace._data.items():
        item_type, dims = _describe(name, item, workspace)
        items.append(WorkspaceItem(name=name, type=item_type, dims=dims))
    return WorkspaceListing(items=items, session_id=session.session_id)


@router.get("/items/{name}", response_model=ItemDetail)
def get_item(name: str, session: Session = Depends(current_session)) -> ItemDetail:
    """Metadata for the inspector panel."""
    workspace = session.workspace
    item = workspace._data.get(name)
    if item is None:
        raise HTTPException(status_code=404, detail=f"'{name}' is not in this workspace.")

    item_type, dims = _describe(name, item, workspace)
    raw_type = item.get("type", "Unknown")

    metadata: dict = {}
    if raw_type == "spectroscopy_tree":
        tensor = _pull_raw(workspace, name)
        if tensor is not None:
            metadata = {
                "Data Type": tensor.data_type,
                "Dimensions": " x ".join(str(n) for n in tensor.value.shape),
                "Axes": ", ".join(
                    f"{label} ({unit})" if unit else label
                    for label, unit in zip(tensor.labels, tensor.units)
                ),
                **{str(k): str(v) for k, v in tensor.metadata.items()},
            }
    elif raw_type == "band_structure":
        metadata = {"High Symmetry Labels": ", ".join(map(str, item.get("labels", [])))}
    elif raw_type == "crystal_structure":
        structure = item.get("structure")
        if structure is not None:
            lattice = structure.lattice
            metadata = {
                "Formula": structure.composition.reduced_formula,
                "Space Group": CrystalEngine.get_symmetry_info(structure)["spacegroup"],
                "Sites": str(len(structure)),
                "a, b, c (A)": f"{lattice.a:.4f}, {lattice.b:.4f}, {lattice.c:.4f}",
                "alpha, beta, gamma": f"{lattice.alpha:.2f}, {lattice.beta:.2f}, {lattice.gamma:.2f}",
                "Volume (A^3)": f"{lattice.volume:.3f}",
            }

    return ItemDetail(
        name=name,
        type=item_type,
        dims=dims,
        metadata=metadata,
        viewable=raw_type in VIEWABLE_TYPES,
    )


@router.post("/demo", response_model=WorkspaceListing)
def seed_demo(
    request: DemoSeedRequest,
    session: Session = Depends(current_session),
) -> WorkspaceListing:
    """Populate an empty session with a synthetic cube.

    Development scaffolding so the UI can be exercised before the beamline
    loaders are wired up. This is a test pattern, not a physics model; the
    ARPES slice replaces it with real loaders from `core/io/`.
    """
    energy = np.linspace(-2.0, 0.5, request.n_energy)
    kx = np.linspace(-2.0, 2.0, request.n_kx)
    ky = np.linspace(-2.0, 2.0, request.n_ky)

    e_grid, kx_grid, ky_grid = np.meshgrid(energy, kx, ky, indexing="ij")
    value = np.exp(-((kx_grid**2 + ky_grid**2) * 2.0 + (e_grid + 0.75) ** 2 * 8.0))

    tensor = TensorData(
        value=value,
        axes=[energy, kx, ky],
        labels=["Energy", "kx (Slit)", "ky (Deflection)"],
        units=["eV", "1/A", "1/A"],
        data_type="Demo Fixture",
        metadata={"Source": "Development fixture", "Session": session.session_id[:8]},
    )
    session.workspace.push_spectroscopy_data(request.name, tensor)

    return list_items(session=session)
