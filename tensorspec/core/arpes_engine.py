import numpy as np
from tensorspec.core.arpes.one_step.chinook_wrapper import ChinookWrapper

class ARPESEngineRouter:
    """
    MAIN ROUTER for all ARPES calculations.
    Follows Rule 8: Never lets solver parameters bleed into UI.
    Routes GUI configs to the designated physics submodules (A, B1, B2, B3).
    """
    def __init__(self):
        # Initialize the available simulation engines
        self.engine_b1 = ChinookWrapper()
        
        # Placeholders for future hierarchical engine implementations
        # self.engine_a = ThreeStepWrapper()     
        # self.engine_b2 = KMapWrapper()         
        # self.engine_b3 = KKRWrapper()          

    def run_simulation(self, model_choice, crystal_data, experiment_kwargs):
        """
        Routes the simulation request based on the model_choice selected in the UI.
        
        Args:
            model_choice (str): 'A', 'B1', 'B2', or 'B3'
            crystal_data (dict): The perfect tb_dict exported from the DFT engine.
            experiment_kwargs (dict): hv, polarization, temperature, k_bounds, resolution.
            
        Returns:
            dict: Simulation results containing the broadened intensity matrix.
        """
        if model_choice == 'B1':
            # 1. Feed the raw DFT dictionary directly into the Chinook wrapper
            self.engine_b1.build_model(crystal_data)
            
            # 2. Run the Fermi's Golden Rule ARPES simulation
            return self.engine_b1.run_simulation(experiment_kwargs)
            
        elif model_choice == 'A':
            raise NotImplementedError("Three-Step Model (Option A) is not yet implemented.")
        elif model_choice == 'B2':
            raise NotImplementedError("kMap Tomography (Option B2) is not yet implemented.")
        elif model_choice == 'B3':
            raise NotImplementedError("SPR-KKR Multiple Scattering (Option B3) is not yet implemented.")
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")