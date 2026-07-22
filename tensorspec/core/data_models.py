import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class TensorData:
    """
    The universal, N-Dimensional data container for TensorSpec.
    This makes the Data Viewer completely agnostic to the physics (ARPES, PEEM, XMCD).
    """
    value: np.ndarray              # The N-dimensional data block (e.g., 3D intensity matrix)
    axes: List[np.ndarray]         # 1D arrays defining the coordinates for each dimension
    labels: List[str]              # Names of each axis (e.g., ["Energy", "Slit Angle", "Deflection"])
    units: List[str]               # Units for each axis (e.g., ["eV", "deg", "deg"])
    data_type: str                 # e.g., "Simulated ARPES", "Experimental PEEM"
    metadata: Dict[str, Any] = field(default_factory=dict) # Flexible dictionary for experimental params

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the primary data matrix."""
        return self.value.ndim
        
    def __post_init__(self):
        """Safety check to ensure the math aligns with the labels."""
        if len(self.axes) != self.ndim:
            raise ValueError(f"Data has {self.ndim} dimensions, but {len(self.axes)} axes were provided.")
        if len(self.labels) != self.ndim or len(self.units) != self.ndim:
            raise ValueError("Labels and units lists must match the number of dimensions.")