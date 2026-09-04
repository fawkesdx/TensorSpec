"""Legacy Maestro dict ↔ TensorData conversions for ML workers and viewer."""
import numpy as np

from tensorspec.core.data_models import TensorData


def tensor_to_ml_dict(td: TensorData) -> dict:
    """Expose TensorData through the legacy mapping consumed by ML workers."""
    data = {
        "value": td.value,
        "kind": td.data_type,
    }
    for label, axis in zip(td.labels, td.axes):
        normalized = label.strip().casefold()
        if normalized == "energy":
            key = "E"
        elif normalized == "angle" or "slit angle" in normalized:
            key = "angle"
        elif normalized == "y":
            key = "y"
        elif normalized == "x":
            key = "x"
        elif normalized.startswith("defl"):
            key = "defl"
        else:
            key = label
        data[key] = axis
    layers = td.metadata.get("layers") or {}
    if isinstance(layers, dict):
        data.update(layers)
    return data


# Monolith alias kept for callers migrating from maestroai_gui.
_tensor_to_ml_dict = tensor_to_ml_dict


def convert_to_tensor_data(data):
    """Convert legacy Maestro workspace dictionaries to TensorData."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise TypeError("convert_to_tensor_data only accepts legacy dictionaries")
    layers = {}
    for k, v in data.items():
        if (
            k.startswith("Labels_")
            or k.startswith("domains_")
            or k.startswith("probs_")
            or k.startswith("embeddings_")
        ):
            layers[k] = v

    axes = [
        data.get("E", np.array([0])),
        data.get("angle", np.array([0])),
        data.get("y", np.array([0])),
        data.get("x", np.array([0])),
    ]
    labels = ["Energy", "Angle", "Y", "X"]
    units = ["eV", "deg", "mm", "mm"]
    return TensorData(
        value=data["value"],
        axes=axes,
        labels=labels,
        units=units,
        data_type=data.get("kind", "Maestro Data"),
        metadata={"layers": layers},
    )
