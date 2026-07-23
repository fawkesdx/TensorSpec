import datetime
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
            attrs={"long_name": tensor_data.data_type}
        )
        
        # 3. Package into a Dataset with full metadata attached
        ds_raw = xr.Dataset({"data": da})
        ds_raw.attrs.update(tensor_data.metadata)
        
        # 4. Construct the strict hierarchical tree
        tree = DataTree.from_dict({
            "/": xr.Dataset(attrs={"dataset_name": name, "creation_time": datetime.datetime.now().isoformat()}),
            "/raw": ds_raw,                          # Immutable raw data
            "/processed": xr.Dataset(),              # Calibrated/transformed data
            "/analysis": xr.Dataset(),               # Mathematical fits (EDC/MDC)
            "/history": xr.Dataset(attrs={"log": [f"[{datetime.datetime.now().time()}] Initialized from {tensor_data.data_type}"]})
        })
        
        return tree