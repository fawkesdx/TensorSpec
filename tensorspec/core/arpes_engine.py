import numpy as np
from tensorspec.core.arpes.one_step.chinook_wrapper import ChinookWrapper
from tensorspec.core.arpes.three_step import ThreeStepWrapper

class ARPESEngineRouter:
    """
    MAIN ROUTER for all ARPES calculations.
    Follows Rule 8: Never lets solver parameters bleed into UI.
    Routes GUI configs to the designated physics submodules (A, B1, B2, B3).
    """
    def __init__(self):
        # Initialize the available simulation engines
        self.engine_b1 = ChinookWrapper()
        self.engine_a = ThreeStepWrapper()
        
        # Placeholders for future hierarchical engine implementations      
        # self.engine_b2 = KMapWrapper()         
        # self.engine_b3 = KKRWrapper()     

    def run_simulation(self, model_choice, crystal_data, experiment_kwargs):
        # 1. Fetch the Fermi level that the DFT suite already calculated
        # Assuming the band_data object passed from DFT holds this info
        e_fermi = crystal_data.get('e_fermi', 0.0) 
        
        if model_choice == 'B1':
            # 2. Inject the Fermi shift into the Chinook model BEFORE solving
            experiment_kwargs['e_fermi'] = e_fermi
            self.engine_b1.build_model(crystal_data)
            return self.engine_b1.run_simulation(experiment_kwargs)
            
        elif model_choice == 'A':
            # Route to the Phenomenological Three-Step Model (Moser framework)
            return self.engine_a.run_simulation(crystal_data, experiment_kwargs)
        elif model_choice == 'B2':
            raise NotImplementedError("kMap Tomography (Option B2) is not yet implemented.")
        elif model_choice == 'B3':
            raise NotImplementedError("SPR-KKR Multiple Scattering (Option B3) is not yet implemented.")
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")