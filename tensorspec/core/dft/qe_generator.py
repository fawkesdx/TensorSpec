import os
import re
import shutil
from pymatgen.core import Structure

class QEInputGenerator:
    """
    Core physics/math engine for generating Quantum Espresso (pw.x) and Wannier90 inputs.
    Zero GUI dependencies. Operates entirely independently of local executables.
    """
    def __init__(self, structure: Structure):
        self.structure = structure
        self.prefix = "tensorspec_run"
        self.app_pseudo_dir = "./pseudo" 
        
    def _generate_atomic_species(self, out_dir: str) -> str:
        """Extracts unique elements, finds their UPF files, and copies them to the run directory."""
        species = []
        os.makedirs(self.app_pseudo_dir, exist_ok=True)
        
        run_pseudo_dir = os.path.join(out_dir, "pseudo")
        os.makedirs(run_pseudo_dir, exist_ok=True)
        
        for el in self.structure.composition.elements:
            mass = el.atomic_mass
            symbol = el.symbol
            
            pseudo_name = None
            pattern = re.compile(rf"^{symbol}[._].*\.upf$", re.IGNORECASE)
            
            for file in os.listdir(self.app_pseudo_dir):
                if pattern.match(file):
                    pseudo_name = file
                    break
                    
            if pseudo_name is None:
                raise FileNotFoundError(
                    f"Missing pseudopotential for {symbol}! "
                    f"Please place a {symbol} UPF file in '{self.app_pseudo_dir}'."
                )
            
            shutil.copy2(os.path.join(self.app_pseudo_dir, pseudo_name), 
                         os.path.join(run_pseudo_dir, pseudo_name))
                
            species.append(f" {symbol}  {float(mass):.4f}  {pseudo_name}")
            
        return "\n".join(species)

    def _generate_explicit_kpoints(self, kmesh: tuple) -> list:
        """Generates the full unreduced k-point grid to 10 decimal places."""
        kpts = []
        for x in range(kmesh[0]):
            for y in range(kmesh[1]):
                for z in range(kmesh[2]):
                    kpts.append(f"  {x/kmesh[0]:.10f}  {y/kmesh[1]:.10f}  {z/kmesh[2]:.10f}")
        return kpts

    def write_scf_input(self, out_dir: str, ecutwfc: float = 60.0, ecutrho: float = 240.0, kmesh: tuple = (6, 6, 6), use_soc: bool = False):
        """Generates the main self-consistent field (SCF) input file."""
        os.makedirs(out_dir, exist_ok=True)
        scf_path = os.path.join(out_dir, "scf.in")
        abs_out = os.path.abspath(os.path.join(out_dir, "out")) + "/"
        
        ibrav = 0 
        nat = len(self.structure)
        ntyp = len(self.structure.composition.elements)

        atomic_species_str = self._generate_atomic_species(out_dir)
        
        # Use the UI toggle to inject SOC
        soc_flags = "\n  noncolin = .true.\n  lspinorb = .true." if use_soc else ""

        scf_content = f"""&CONTROL
  calculation = 'scf'
  prefix = '{self.prefix}'
  outdir = '{abs_out}'
  pseudo_dir = './pseudo/'
  wf_collect = .true.
/
&SYSTEM
  ibrav = {ibrav}
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = {ecutwfc}
  ecutrho = {ecutrho}
  occupations = 'smearing'
  smearing = 'marzari-vanderbilt'
  degauss = 0.01{soc_flags}
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.7
/
ATOMIC_SPECIES
{atomic_species_str}

{self._generate_cell_parameters()}

{self._generate_atomic_positions()}

K_POINTS {{automatic}}
  {kmesh[0]} {kmesh[1]} {kmesh[2]}  0 0 0
"""
        with open(scf_path, "w") as f:
            f.write(scf_content)
        return scf_path

    def write_nscf_input(self, out_dir: str, ecutwfc: float = 60.0, ecutrho: float = 240.0, kmesh: tuple = (6, 6, 6), nbnd: int = 12, use_soc: bool = False):
        """Generates the non-self-consistent field (NSCF) input file with explicit k-points."""
        nscf_path = os.path.join(out_dir, "nscf.in")
        abs_out = os.path.abspath(os.path.join(out_dir, "out")) + "/"
        
        ibrav = 0  
        nat = len(self.structure)
        ntyp = len(self.structure.composition.elements)
        
        atomic_species_str = self._generate_atomic_species(out_dir)
        
        kpts = self._generate_explicit_kpoints(kmesh)
        kpts_qe = "\n".join([f"{k}  1.0" for k in kpts])
        
        # Use the UI toggle to inject SOC
        soc_flags = "\n  noncolin = .true.\n  lspinorb = .true." if use_soc else ""

        nscf_content = f"""&CONTROL
  calculation = 'nscf'
  prefix = '{self.prefix}'
  outdir = '{abs_out}'
  pseudo_dir = './pseudo/'
  wf_collect = .true.
/
&SYSTEM
  ibrav = {ibrav}
  nat = {nat}
  ntyp = {ntyp}
  nbnd = {nbnd}
  nosym = .true.
  noinv = .true.
  ecutwfc = {ecutwfc}
  ecutrho = {ecutrho}
  occupations = 'smearing'
  smearing = 'marzari-vanderbilt'
  degauss = 0.01{soc_flags}
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.7
/
ATOMIC_SPECIES
{atomic_species_str}

{self._generate_cell_parameters()}

{self._generate_atomic_positions()}

K_POINTS {{crystal}}
{len(kpts)}
{kpts_qe}
"""
        with open(nscf_path, "w") as f:
            f.write(nscf_content)
        return nscf_path

    def write_wannier90_input(self, out_dir: str, kmesh: tuple = (6, 6, 6), num_wann: int = 12, use_soc: bool = False):
        """Generates a base wannier90.win file for extracting the tight binding Hamiltonian."""
        win_path = os.path.join(out_dir, "wannier90.win")
        
        kpts = self._generate_explicit_kpoints(kmesh)
        kpts_win = "\n".join(kpts)
        
        proj_lines = []
        for el in self.structure.composition.elements:
            if el.is_transition_metal or el.number > 30:
                proj_lines.append(f"{el.symbol}:s;p;d")
            else:
                proj_lines.append(f"{el.symbol}:s;p")
        proj_string = "\n".join(proj_lines)
        
        # Use the UI toggle to inject SOC
        spinor_str = "spinors = true\n" if use_soc else ""
        
        # change back the num_iter = 100 and apply wanniertool symmetrizer after that before chinook!!
        win_content = f"""num_wann = {num_wann}
num_iter = 0
num_print_cycles = 10
write_hr = true
write_xyz = true
use_ws_distance = true
{spinor_str}
begin projections
{proj_string}
end projections

begin unit_cell_cart
{chr(10).join(["  " + "  ".join([f"{v:.6f}" for v in row]) for row in self.structure.lattice.matrix])}
end unit_cell_cart

begin atoms_frac
{chr(10).join([f" {site.species_string}  {site.frac_coords[0]:.6f}  {site.frac_coords[1]:.6f}  {site.frac_coords[2]:.6f}" for site in self.structure])}
end atoms_frac

mp_grid = {kmesh[0]} {kmesh[1]} {kmesh[2]}

begin kpoints
{kpts_win}
end kpoints
"""
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        try:
            kpath_obj = HighSymmKpath(self.structure)
            kpts_dict = kpath_obj.kpath['kpoints']
            path_segments = kpath_obj.kpath['path']
            
            path_lines = []
            for segment in path_segments:
                for i in range(len(segment) - 1):
                    l1, l2 = segment[i], segment[i+1]
                    p1, p2 = kpts_dict[l1], kpts_dict[l2]
                    l1_str = "G" if "Gamma" in l1 or l1 == "\\Gamma" else l1
                    l2_str = "G" if "Gamma" in l2 or l2 == "\\Gamma" else l2
                    path_lines.append(f"{l1_str} {p1[0]:.5f} {p1[1]:.5f} {p1[2]:.5f}  {l2_str} {p2[0]:.5f} {p2[1]:.5f} {p2[2]:.5f}")
                    
            win_content += "\nbands_plot = true\n"
            win_content += "bands_num_points = 100\n"
            win_content += "begin kpoint_path\n"
            win_content += "\n".join(path_lines)
            win_content += "\nend kpoint_path\n"
        except Exception as e:
            print(f"Could not auto-generate W90 k-path: {e}")

        with open(win_path, "w") as f:
            f.write(win_content)
        return win_path

    def write_pw2wan_input(self, out_dir: str):
        """Generates the bridge input file for pw2wannier90.x."""
        pw2wan_path = os.path.join(out_dir, "pw2wan.in")
        abs_out = os.path.abspath(os.path.join(out_dir, "out")) + "/"
        
        content = f"""&inputpp
  outdir = '{abs_out}'
  prefix = '{self.prefix}'
  seedname = 'wannier90'
  write_mmn = .true.
  write_amn = .true.
  write_unk = .false.
/
"""
        with open(pw2wan_path, "w") as f:
            f.write(content)
        return pw2wan_path

    def _generate_cell_parameters(self) -> str:
        """Extracts the lattice matrix."""
        params = ["CELL_PARAMETERS {angstrom}"]
        for row in self.structure.lattice.matrix:
            params.append("  " + "  ".join([f"{v:.6f}" for v in row]))
        return "\n".join(params)

    def _generate_atomic_positions(self) -> str:
        """Converts PyMatgen fractional coordinates to QE format."""
        positions = ["ATOMIC_POSITIONS {crystal}"]
        for site in self.structure:
            coords = "  ".join([f"{c:.6f}" for c in site.frac_coords])
            positions.append(f" {site.species_string}  {coords}")
        return "\n".join(positions)
    
    def _detect_soc(self) -> bool:
        """Detects if any provided pseudopotential is fully relativistic (SOC)."""
        for el in self.structure.composition.elements:
            symbol = el.symbol
            pattern = re.compile(rf"^{symbol}[._].*\.upf$", re.IGNORECASE)
            for file in os.listdir(self.app_pseudo_dir):
                if pattern.match(file):
                    if "rel" in file.lower() or "fr" in file.lower():
                        return True
        return False