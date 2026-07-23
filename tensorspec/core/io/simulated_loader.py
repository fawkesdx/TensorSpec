import numpy as np
from pathlib import Path
from tensorspec.core.data_models import TensorData

class SimulatedARPESLoader:
    """Loader for TensorSpec's native .npz simulated ARPES datasets."""
    
    @staticmethod
    def load(filepath: str) -> TensorData:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find simulated data at {filepath}")
            
        with np.load(path, allow_pickle=True) as data:
            intensity = data['intensity']  # Original shape: (kx, ky, E)
            
            # Transpose (kx, ky, E) -> (E, kx, ky) to match the Data Viewer's expected axis order
            value_matrix = np.transpose(intensity, (2, 0, 1))
            
            metadata = data['metadata'].item() if 'metadata' in data else {}
            
            return TensorData(
                value=value_matrix,
                axes=[data['E'], data['kx'], data['ky']],
                labels=["Energy", "kx (Slit)", "ky (Deflection)"],
                units=["eV", "1/A", "1/A"],
                data_type="Simulated ARPES Matrix Elements",
                metadata=metadata
            )