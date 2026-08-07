"""ARPES viewer endpoints: slice tensors and extract curves.

All arithmetic is delegated to `core.tensor_ops`; this module only validates
requests and frames the results for transport.

Slices are returned as binary rather than JSON. A 900x900 plane is 810,000
numbers, which costs roughly 8 MB as JSON text and takes noticeable time to
parse; the same plane is 3.2 MB as raw float32 and lands directly in a typed
array the canvas can read. See `_pack_plane` for the framing.
"""
from __future__ import annotations

import json
import struct

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response

from tensorspec.core import tensor_ops as ops
from tensorspec.web.server.schemas import (
    AxisInfo,
    Curve,
    ProfileRequest,
    ProfileResponse,
    SliceRequest,
    TensorAxes,
)
from tensorspec.web.server.session import Session, current_session

router = APIRouter(prefix="/api/arpes", tags=["arpes"])


def _require_tensor(session: Session, name: str):
    tensor = session.workspace.pull_tensor_data(name)
    if tensor is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not spectroscopy data in this session.",
        )
    return tensor


def _validate_axes(tensor, x_idx: int, y_idx: int) -> None:
    if x_idx == y_idx:
        raise HTTPException(status_code=422, detail="The X and Y axes must differ.")
    for index in (x_idx, y_idx):
        if not 0 <= index < tensor.ndim:
            raise HTTPException(
                status_code=422,
                detail=f"Axis {index} is outside this {tensor.ndim}D tensor.",
            )


def _pack_plane(header: dict, plane: np.ndarray) -> bytes:
    """
    Frames a plane as: uint32 header length, header JSON, then float32 values.

    The header is padded so the numeric payload starts on a 4-byte boundary,
    which lets the browser wrap it in a Float32Array with no copy.
    """
    raw = json.dumps(header).encode("utf-8")
    padding = (-len(raw)) % 4
    raw += b" " * padding
    body = np.ascontiguousarray(plane, dtype="<f4").tobytes()
    return struct.pack("<I", len(raw)) + raw + body


@router.get("/{name}/axes", response_model=TensorAxes)
def get_axes(name: str, session: Session = Depends(current_session)) -> TensorAxes:
    """Dimensions of a tensor, plus a sensible opening view."""
    tensor = _require_tensor(session, name)
    described = ops.describe_axes(tensor)

    # Open on the last two dimensions, which for ARPES data is the detector
    # plane; remaining dimensions start at their midpoint.
    default_y = 0 if tensor.ndim > 1 else 0
    default_x = 1 if tensor.ndim > 1 else 0
    fixed = {
        axis["index"]: axis["size"] // 2
        for axis in described
        if axis["index"] not in (default_x, default_y)
    }

    return TensorAxes(
        name=name,
        data_type=tensor.data_type,
        ndim=tensor.ndim,
        axes=[AxisInfo(**axis) for axis in described],
        default_x=default_x,
        default_y=default_y,
        default_fixed=fixed,
    )


@router.post("/{name}/slice")
def get_slice(
    name: str,
    request: SliceRequest,
    session: Session = Depends(current_session),
) -> Response:
    """A 2D plane, decimated to fit the viewport, as a binary payload."""
    tensor = _require_tensor(session, name)
    _validate_axes(tensor, request.x_idx, request.y_idx)

    result = ops.extract_slice(tensor, request.x_idx, request.y_idx, request.fixed)
    plane, x_axis, y_axis, steps = ops.downsample_plane(
        result["values"], result["x_axis"], result["y_axis"], request.max_points
    )

    header = {
        "shape": [int(plane.shape[0]), int(plane.shape[1])],
        "x_axis": [float(v) for v in x_axis],
        "y_axis": [float(v) for v in y_axis],
        "extent": [float(v) for v in result["extent"]],
        "vmin": result["vmin"],
        "vmax": result["vmax"],
        "stride": [int(steps[0]), int(steps[1])],
        "x_label": tensor.labels[request.x_idx],
        "y_label": tensor.labels[request.y_idx],
        "x_unit": tensor.units[request.x_idx],
        "y_unit": tensor.units[request.y_idx],
        "full_shape": [int(result["values"].shape[0]), int(result["values"].shape[1])],
    }
    return Response(content=_pack_plane(header, plane), media_type="application/octet-stream")


@router.post("/{name}/profiles", response_model=ProfileResponse)
def get_profiles(
    name: str,
    request: ProfileRequest,
    session: Session = Depends(current_session),
) -> ProfileResponse:
    """
    The curves through the crosshair.

    With energy on Y, the `y` curve is the EDC and `x` is the MDC; the names
    stay geometric because the user chooses which physical axis goes where.
    """
    tensor = _require_tensor(session, name)
    _validate_axes(tensor, request.x_idx, request.y_idx)

    if request.ortho_idx is not None:
        if not 0 <= request.ortho_idx < tensor.ndim:
            raise HTTPException(status_code=422, detail="Orthogonal axis is out of range.")
        if request.ortho_idx in (request.x_idx, request.y_idx):
            raise HTTPException(
                status_code=422,
                detail="The orthogonal axis must differ from the displayed axes.",
            )

    result = ops.extract_slice(tensor, request.x_idx, request.y_idx, request.fixed)
    plane = result["values"]

    x_center = min(request.x_center, plane.shape[1] - 1)
    y_center = min(request.y_center, plane.shape[0] - 1)
    x_bounds = ops.integration_bounds(x_center, request.dx, plane.shape[1])
    y_bounds = ops.integration_bounds(y_center, request.dy, plane.shape[0])

    profiles = ops.extract_profiles(plane, x_bounds, y_bounds, request.mode)

    ortho_curve = None
    if request.ortho_idx is not None:
        ortho = ops.extract_orthogonal_profile(
            tensor, request.ortho_idx, request.x_idx, request.y_idx,
            x_bounds, y_bounds, request.fixed, request.mode,
        )
        ortho_curve = Curve(
            axis=[float(v) for v in ortho["axis"]],
            values=[float(v) for v in ortho["values"]],
            label=tensor.labels[request.ortho_idx],
            unit=tensor.units[request.ortho_idx],
        )

    return ProfileResponse(
        x=Curve(
            axis=[float(v) for v in result["x_axis"]],
            values=[float(v) for v in profiles["x"]],
            label=tensor.labels[request.x_idx],
            unit=tensor.units[request.x_idx],
        ),
        y=Curve(
            axis=[float(v) for v in result["y_axis"]],
            values=[float(v) for v in profiles["y"]],
            label=tensor.labels[request.y_idx],
            unit=tensor.units[request.y_idx],
        ),
        ortho=ortho_curve,
        window={
            "x1": x_bounds[0], "x2": x_bounds[1],
            "y1": y_bounds[0], "y2": y_bounds[1],
        },
        mode=request.mode,
    )
