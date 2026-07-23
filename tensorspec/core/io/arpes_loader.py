import os
from .loaders.mock_data_loader import MockDataLoader
from .loaders.maestro_loader import MaestroLoader
from tensorspec.core.data_models import TensorData

class ARPESLoader:
    """
    Main agnostic IO Manager.
    Passes the file through loaders, then packages it into the universal TensorData format.
    """
    @classmethod
    def load(cls, filepath):
        filepath_str = str(filepath)
        filename = os.path.basename(filepath_str)
        
        available_loaders = [
            MockDataLoader, 
            MaestroLoader
        ]
        
        for LoaderClass in available_loaders:
            try:
                loader = LoaderClass(filepath_str)
                raw_dict = loader.load()
                
                # Extract axes and labels
                axes_dict = raw_dict.get('axes', {})
                labels = list(axes_dict.keys())
                axes = list(axes_dict.values())
                
                # Dynamically assign units based on axis names
                units = ["eV" if "eV" in l else "deg" if "deg" in l else "a.u." for l in labels]
                
                # Package it perfectly into your central architecture
                return TensorData(
                    value=raw_dict['data'],
                    axes=axes,
                    labels=labels,
                    units=units,
                    data_type=raw_dict.get('mode', 'ARPES'),
                    metadata=raw_dict.get('metadata', {})
                )
                
            except ValueError:
                # Signature didn't match, move to next loader
                continue
            except Exception as e:
                print(f"Warning: {LoaderClass.__name__} encountered an error: {e}")
                continue
                
        raise ValueError(f"Could not load '{filename}': No matching facility format found.")