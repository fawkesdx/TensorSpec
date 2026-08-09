import datetime

import numpy as np
import xarray as xr
from xarray import DataTree

from tensorspec.core.data_models import TensorData

class DataTreeBuilder:
    """
    Factory class to construct standardized xarray.DataTree hierarchical 
    structures from agnostic TensorData objects.
    """

    @staticmethod
    def build_from_tensor(name: str, tensor_data: TensorData) -> DataTree:
        """
        Wraps incoming beamline or simulated data into the NeXus/HDF5 aligned hierarchy:
        /raw, /processed, /analysis, /history.
        """
        # 1. Map labels and axes to xarray coordinates
        coords = {
            label: (label, ax, {"units": unit})
            for label, ax, unit in zip(tensor_data.labels, tensor_data.axes, tensor_data.units)
        }

        # 2. Build the primary DataArray
        da = xr.DataArray(
            data=tensor_data.value,
            coords=coords,
            dims=tensor_data.labels,
            name="intensity",
            attrs={"long_name": tensor_data.data_type},
        )

        # 3. Package into a Dataset with metadata (motors go under /raw/motors)
        metadata = dict(tensor_data.metadata or {})
        motors = metadata.pop("motors", None)
        ds_raw = xr.Dataset({"data": da})
        ds_raw.attrs.update(metadata)

        nodes = {
            "/": xr.Dataset(
                attrs={
                    "dataset_name": name,
                    "creation_time": datetime.datetime.now().isoformat(),
                }
            ),
            "/raw": ds_raw,
            "/processed": xr.Dataset(),
            "/analysis": xr.Dataset(),
            "/history": xr.Dataset(
                attrs={
                    "log": [
                        f"[{datetime.datetime.now().time()}] Initialized from {tensor_data.data_type}"
                    ]
                }
            ),
        }

        if isinstance(motors, dict) and motors:
            motor_vars = {
                key: (("point",), np.asarray(values, dtype=float))
                for key, values in motors.items()
            }
            nodes["/raw/motors"] = xr.Dataset(motor_vars)

        return DataTree.from_dict(nodes)

    @staticmethod
    def dataset_from_tensor(tensor_data: TensorData) -> xr.Dataset:
        """Build a single Dataset (no tree) for writing into /processed."""
        coords = {
            label: (label, ax, {"units": unit})
            for label, ax, unit in zip(tensor_data.labels, tensor_data.axes, tensor_data.units)
        }
        da = xr.DataArray(
            data=tensor_data.value,
            coords=coords,
            dims=tensor_data.labels,
            name="intensity",
            attrs={"long_name": tensor_data.data_type},
        )
        metadata = dict(tensor_data.metadata or {})
        metadata.pop("motors", None)
        ds = xr.Dataset({"data": da})
        ds.attrs.update(metadata)
        return ds

    @staticmethod
    def merge_raw_attrs(tree: DataTree, attrs: dict) -> DataTree:
        """Merge metadata into ``/raw`` without rebuilding its intensity cube."""
        tree["raw"].ds.attrs.update(attrs)
        return tree

    @staticmethod
    def write_processed(tree: DataTree, tensor_data: TensorData) -> DataTree:
        """Replace /processed on an existing tree and append a history line."""
        ds = DataTreeBuilder.dataset_from_tensor(tensor_data)
        tree["processed"] = ds

        history_node = tree["history"]
        history = history_node.to_dataset() if hasattr(history_node, "to_dataset") else history_node.ds
        log = list(history.attrs.get("log") or [])
        log.append(
            f"[{datetime.datetime.now().time()}] Wrote /processed ({tensor_data.data_type})"
        )
        history = history.copy()
        history.attrs["log"] = log
        tree["history"] = history
        return tree

    @staticmethod
    def write_analysis(tree: DataTree, node_name: str, dataset: xr.Dataset) -> DataTree:
        """Write ``/analysis/<node_name>`` and append a history line."""
        safe = node_name.strip().strip("/")
        if not safe or "/" in safe:
            raise ValueError("Analysis node name must be a single path segment.")
        tree[f"analysis/{safe}"] = dataset

        history_node = tree["history"]
        history = history_node.to_dataset() if hasattr(history_node, "to_dataset") else history_node.ds
        log = list(history.attrs.get("log") or [])
        log.append(f"[{datetime.datetime.now().time()}] Wrote /analysis/{safe}")
        history = history.copy()
        history.attrs["log"] = log
        tree["history"] = history
        return tree