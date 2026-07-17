import numpy as np

try:
    import chinook.build_lib as build_lib
    import chinook.ARPES_lib as arpes_lib
except ImportError:
    build_lib = None
    arpes_lib = None

class ChinookWrapper:
    """
    Engine B1: Tight-Binding + Free Electron Final State
    Acts as the interface between TensorSpec and the Chinook ARPES simulator.
    Strictly pure math/logic. Zero GUI imports.
    """
    def __init__(self):
        self.tb_model = None
        self.arpes_experiment = None

    def build_model(self, basis_vectors, atoms, orbitals, slater_koster_dict):
        """
        Translates TensorSpec's workspace crystal and orbital data into a Chinook TB model.
        """
        if build_lib is None:
            raise ImportError("Chinook is not installed. Please install it to use Engine B1.")
        
        # 1. Define the Chinook basis (atoms and their fractional coordinates)
        chinook_basis = []
        for i, atom in enumerate(atoms):
            # mapping atom symbol and coords to chinook formatting
            chinook_basis.append([atom['symbol'], atom['coords']])
            
        # 2. Build the tight-binding model using Chinook's dictionary format
        tb_dict = {
            'type': 'tb',
            'a': basis_vectors,
            'atoms': chinook_basis,
            'orbitals': orbitals,  # Expected format: ['W', 'dz2', 'W', 'dx2-y2', ...]
            'hopping': slater_koster_dict 
        }
        
        # Initialize the model
        self.tb_model = build_lib.gen_tb(tb_dict)
        return True

    def run_simulation(self, experiment_kwargs):
        """
        Executes the photoemission calculation (Fermi's Golden Rule).
        
        Args:
            experiment_kwargs (dict): Configuration routed from the GUI (hv, polarization, T, etc.)
        Returns:
            dict: Raw intensity and resolution-broadened intensity matrices.
        """
        if self.tb_model is None:
            raise ValueError("Tight-binding model must be built before running ARPES simulation.")
        
        # Map TensorSpec standard kwargs to Chinook's exact parameter names
        chinook_params = {
            'cube': experiment_kwargs.get('k_bounds'),       # [kx_min, kx_max, ky_min, ky_max, E_min, E_max]
            'resolution': experiment_kwargs.get('resolution', [0.01, 0.01]), # [dk, dE]
            'hv': experiment_kwargs.get('photon_energy', 21.2), # eV
            'pol': experiment_kwargs.get('polarization', np.array([1, 0, 0])), # Vector
            'T': experiment_kwargs.get('temperature', 10),      # Kelvin
            'spin': experiment_kwargs.get('spin_resolve', None),
            'TB': self.tb_model
        }
        
        # Initialize and run the experiment
        self.arpes_experiment = arpes_lib.experiment(chinook_params)
        self.arpes_experiment.run_ARPES()
        
        # Extract the results (Ig is the broadened intensity heatmap)
        results = {
            'k_points': self.arpes_experiment.k,
            'energy': self.arpes_experiment.E,
            'intensity_raw': self.arpes_experiment.I,
            'intensity_broadened': self.arpes_experiment.Ig
        }
        
        return results