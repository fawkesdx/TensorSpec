import os
import numpy as np
from pymatgen.core import Structure

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:
    AseAtomsAdaptor = None


# Conversion constant: 1 Angstrom to Bohr (atomic units)
ANG_TO_BOHR = 1.8897261254578281


class SPRKKRInputGenerator:
    """
    Core physics engine for generating SPR-KKR SCF input (.inp) and starting potential (.pot) files.
    Zero GUI dependencies. Operates independently of local executables.
    """

    def __init__(self, structure: Structure = None):
        if structure is not None and not isinstance(structure, Structure):
            raise ValueError("Invalid structure passed to SPRKKRInputGenerator. Expected pymatgen.core.Structure.")
        self.structure = structure
        self.dataset = "scf"
        self.prefix = "scf"
        self.pot_filename = "scf.pot"
        self.inp_filename = "scf.inp"

    def to_ase_atoms(self):
        """Converts pymatgen Structure to ASE Atoms object if adapter is available."""
        if AseAtomsAdaptor is not None:
            try:
                return AseAtomsAdaptor.get_atoms(self.structure)
            except Exception:
                pass
        return None

    def _rel_to_irel(self, rel_mode: str = "Fully Relativistic (Dirac)", use_soc: bool = True) -> int:
        """Maps relativity description / SOC flag to SPR-KKR IREL flag (0, 1, 2, 3)."""
        mode = (rel_mode or "").lower()
        if "non" in mode:
            return 0
        elif "scalar" in mode:
            return 2 if use_soc else 1
        elif "dirac" in mode or "fully" in mode or use_soc:
            return 3
        return 3

    def _get_unique_species(self):
        """Returns unique element symbols and atomic numbers preserve order."""
        unique_elements = []
        for site in self.structure:
            sym = site.specie.symbol
            if sym not in unique_elements:
                unique_elements.append(sym)
        return unique_elements

    def _generate_lattice_info(self):
        """
        Extracts lattice constant ALAT in Bohr and normalized unit cell basis vectors A(1), A(2), A(3).
        """
        lattice = self.structure.lattice
        matrix_ang = lattice.matrix  # shape (3, 3) in Angstroms
        a_ang = lattice.a
        alat_bohr = a_ang * ANG_TO_BOHR

        # Normalized lattice basis vectors in units of alat
        abas = matrix_ang / a_ang

        return alat_bohr, abas

    def generate_inp_content(
        self,
        lmax: int = 3,
        nktab: int = 250,
        ne: int = 30,
        rel_mode: str = "Fully Relativistic (Dirac)",
        use_soc: bool = True,
        niter: int = 100,
        mix: float = 0.2,
        tol: float = 1e-6,
        xc: str = "VWN",
        nonmag: bool = False,
        print_level: int = 0,
    ) -> str:
        """Generates standard SPRKKR scf.inp (Namelist format)."""
        irel = self._rel_to_irel(rel_mode, use_soc)

        inp_lines = [
            "CONTROL",
            f"  DATASET     = {self.dataset}",
            "  ADSI        = SCF",
            f"  POTFIL      = {self.pot_filename}",
            f"  PRINT       = {print_level}",
            "  NONMAG      = F" if not nonmag else "  NONMAG      = T",
            "",
            "MODE",
            f"  IREL        = {irel}",
            "  LLOYD       = F",
            "",
            "TAU",
            "  BZINT       = POINTS",
            f"  NKTAB       = {nktab}",
            f"  LMAX        = {lmax}",
            "",
            "ENERGY",
            "  EMIN        = -0.200000",
            "  EMAX        = 1.000000",
            f"  NE          = {ne}",
            "  EFERMI      = 0.500000",
            "  IMAG        = 0.000000",
            "",
            "SCF",
            f"  NITER       = {niter}",
            f"  MIX         = {mix:.6f}",
            f"  VXC         = {xc}",
            f"  TOL         = {tol:.6E}".replace("E", "D"),
            "  ISTBRY      = 1",
            "",
        ]
        return "\n".join(inp_lines)

    def generate_pot_content(
        self,
        nktab: int = 250,
        ne: int = 30,
        rel_mode: str = "Fully Relativistic (Dirac)",
        use_soc: bool = True,
        mix: float = 0.2,
        tol: float = 1e-6,
        xc: str = "VWN",
        nonmag: bool = False,
    ) -> str:
        """Generates standard SPRKKR scf.pot (Format 7 with SCFSTATUS START)."""
        irel = self._rel_to_irel(rel_mode, use_soc)
        nspin = 1 if nonmag or irel in (0, 1) else 2
        nonmag_str = "T" if nonmag else "F"

        formula = self.structure.composition.reduced_formula
        nq = len(self.structure)
        unique_species = self._get_unique_species()
        nt = len(unique_species)
        nm = nt  # Number of distinct radial meshes

        species_to_type = {sym: idx + 1 for idx, sym in enumerate(unique_species)}

        alat_bohr, abas = self._generate_lattice_info()

        lines = [
            "SPRKKR potential file (first line skipped by Fortran parser)",
            "HEADER    SPR-KKR dataset created by TensorSpec",
            f"TITLE     {formula}",
            f"SYSTEM    {formula}",
            "PACKAGE   SPR-KKR",
            "FORMAT    7 (18.01.2019)",
            "GLOBAL SYSTEM PARAMETER",
            f"NQ        {nq}",
            f"NT        {nt}",
            f"NM        {nm}",
            f"IREL      {irel}",
            f"NSPIN     {nspin}",
            "SCF-INFO",
            "INFO      starting potential generated by TensorSpec",
            "SCFSTATUS START",
            "FULLPOT   F",
            "FINNUC    F",
            "BREITINT  F",
            f"NONMAG    {nonmag_str}",
            "ORBPOL    NONE",
            "EXTFIELD  F",
            "BLCOUPL   F",
            "BEXT      0.00000000000000E+00 0.00000000000000E+00 0.00000000000000E+00",
            "SEMICORE  F",
            "LLOYD     F",
            f"NE        {ne}",
            "IBZINT    2",
            f"NKTAB     {nktab}",
            f"XC-POT    {xc}",
            "SCF-ALG   BROYDEN",
            "SCF-ITER  0",
            f"SCF-MIX   {mix:.14E}",
            f"SCF-TOL   {tol:.14E}",
            "RMSAVV    0.00000000000000E+00",
            "RMSAVB    0.00000000000000E+00",
            "EF        0.50000000000000E+00",
            "VMTZ      0.00000000000000E+00",
            "LATTICE",
            "SYSDIM    3D",
            "SYSTYPE   BULK",
            "BRAVAIS   14  triclinic",
            f"ALAT      {alat_bohr:.14E}",
            f"A(1)      {abas[0,0]:.14E} {abas[0,1]:.14E} {abas[0,2]:.14E}",
            f"A(2)      {abas[1,0]:.14E} {abas[1,1]:.14E} {abas[1,2]:.14E}",
            f"A(3)      {abas[2,0]:.14E} {abas[2,1]:.14E} {abas[2,2]:.14E}",
            "SITES",
            "CARTESIAN F"
        ]

        # SITES: Fractional coordinates
        lines.append("BASSCALE   1.00000000000000E+00  1.00000000000000E+00  1.00000000000000E+00")
        lines.append("   IQ      QBAS(X)           QBAS(Y)           QBAS(Z)")
        for iq, site in enumerate(self.structure, start=1):
            fc = site.frac_coords
            lines.append(f"{iq:10d} {fc[0]:22.14E} {fc[1]:22.14E} {fc[2]:22.14E}")

        # OCCUPATION: IQ, IREFQ, IM, NOQ, (IT, CONC)
        lines.append("OCCUPATION")
        lines.append("   IQ     IREFQ      IMQ      NOQ  ITOQ  CONC")
        for iq, site in enumerate(self.structure, start=1):
            sym = site.specie.symbol
            it = species_to_type[sym]
            im = it
            irefq = im
            lines.append(f"{iq:10d}{irefq:10d}{im:10d}{1:10d}{it:6d}{1.0:8.5f}")

        # REFERENCE SYSTEM FOR TIGHT BINDING MODE
        lines.append("REFERENCE SYSTEM FOR TIGHT BINDING MODE")
        lines.append(f"NREF      {nm}")
        lines.append("   IM     VREF           RMTREF")
        for im in range(1, nm + 1):
            lines.append(f"{im:10d}  0.00000000000000E+00  0.00000000000000E+00")

        # MAGNETISATION DIRECTION
        lines.append("MAGNETISATION DIRECTION")
        lines.append("KMROT     0")
        lines.append("QMVEC     0.00000000000000E+00 0.00000000000000E+00 0.00000000000000E+00")
        lines.append("   IQ       QMTET       QMPHI")
        for iq in range(1, nq + 1):
            lines.append(f"{iq:10d}  0.00000000000000E+00  0.00000000000000E+00")

        # MESH INFORMATION
        lines.append("MESH INFORMATION")
        lines.append("MESH-TYPE EXPONENTIAL")
        lines.append("   IM       R(1)                  DX              JRMT       RMT             JRWS       RWS")
        for im in range(1, nm + 1):
            lines.append(f"{im:5d} 1.00000000000000E-06 1.00000000000000E-02  721 2.00000000000000E+00  721 2.00000000000000E+00")

        # TYPES
        lines.append("TYPES")
        lines.append("   IT  TXT_T           ZT     NCORT     NVALT      NSEMCOR")
        for it, sym in enumerate(unique_species, start=1):
            el = self.structure.composition.elements[0]
            # Match element object
            for candidate in self.structure.composition.elements:
                if candidate.symbol == sym:
                    el = candidate
                    break
            z_num = el.number
            # Standard valence approximation
            nval = int(el.group) if el.group is not None else 4
            ncort = max(0, z_num - nval)
            lines.append(f"{it:5d}     {sym:<8s}{z_num:10d}{ncort:10d}{nval:10d}{0:10d}")

        return "\n".join(lines) + "\n"

    def write_scf_input(
        self,
        out_dir: str,
        lmax: int = 3,
        nktab: int = 250,
        ne: int = 30,
        rel_mode: str = "Fully Relativistic (Dirac)",
        use_soc: bool = True,
        niter: int = 100,
        mix: float = 0.2,
        tol: float = 1e-6,
        xc: str = "VWN",
        nonmag: bool = False,
        print_level: int = 0,
    ) -> str:
        """
        Generates scf.inp and scf.pot in out_dir for SPRKKR calculation.
        Returns the path to scf.inp.
        """
        os.makedirs(out_dir, exist_ok=True)

        inp_path = os.path.join(out_dir, self.inp_filename)
        pot_path = os.path.join(out_dir, self.pot_filename)

        inp_content = self.generate_inp_content(
            lmax=lmax,
            nktab=nktab,
            ne=ne,
            rel_mode=rel_mode,
            use_soc=use_soc,
            niter=niter,
            mix=mix,
            tol=tol,
            xc=xc,
            nonmag=nonmag,
            print_level=print_level,
        )

        pot_content = self.generate_pot_content(
            nktab=nktab,
            ne=ne,
            rel_mode=rel_mode,
            use_soc=use_soc,
            mix=mix,
            tol=tol,
            xc=xc,
            nonmag=nonmag,
        )

        with open(inp_path, "w") as f:
            f.write(inp_content)

        with open(pot_path, "w") as f:
            f.write(pot_content)

        return inp_path

    def write_pot_file(self, out_dir: str, **kwargs) -> str:
        """Writes scf.pot separately."""
        os.makedirs(out_dir, exist_ok=True)
        pot_path = os.path.join(out_dir, self.pot_filename)
        pot_content = self.generate_pot_content(**kwargs)
        with open(pot_path, "w") as f:
            f.write(pot_content)
        return pot_path

    def write_inp_file(self, out_dir: str, **kwargs) -> str:
        """Writes scf.inp separately."""
        os.makedirs(out_dir, exist_ok=True)
        inp_path = os.path.join(out_dir, self.inp_filename)
        inp_content = self.generate_inp_content(**kwargs)
        with open(inp_path, "w") as f:
            f.write(inp_content)
        return inp_path

    def generate_arpes_inp_content(
        self,
        task: str = "ARPES",
        ne: int = 300,
        e_min: float = -10.0,
        e_max: float = 2.0,
        ephot: float = 21.2,
        temp: float = 10.0,
        workf: float = 4.5,
        polar: str = "p",
        hkl: tuple = (0, 0, 1),
    ) -> str:
        polar_map = {
            "p-pol": "P",
            "s-pol": "S",
            "CR": "C+",
            "CL": "C-",
            "Arbitrary": "P"
        }
        polar_code = "P"
        for k, v in polar_map.items():
            if k in polar:
                polar_code = v
                break

        inp_lines = [
            "CONTROL",
            "  DATASET     = sys",
            f"  ADSI        = {task}",
            "  POTFIL      = scf.pot_new",
            "  PRINT       = 0",
            "",
            "MODE",
            "  IREL        = 3",
            "",
            "TAU",
            "  BZINT       = POINTS",
            "  NKTAB       = 250",
            "",
            "ENERGY",
            f"  EMIN        = {e_min:.5f}",
            f"  EMAX        = {e_max:.5f}",
            f"  NE          = {ne}",
            "  EFERMI      = 0.500000",
            "  IMAG        = 0.010000",
            "",
            f"TASK {task}",
            "  STRVER      = 1",
            "  IQ_SURF     = 1",
            f"  MILLER_HKL  = {{{int(hkl[0])} {int(hkl[1])} {int(hkl[2])}}}",
            "  THETA       = 45.0",
            "  PHI         = 0.0",
            f"  EPHOT       = {ephot:.2f}",
            f"  POLAR       = {polar_code}",
            f"  TEMP        = {temp:.2f}",
            f"  WORKF       = {workf:.2f}",
            "",
        ]
        return "\n".join(inp_lines)

    def write_arpes_input(self, out_dir: str, **kwargs) -> str:
        """Writes sys.inp in out_dir for SPRKKR Spectroscopy calculation."""
        os.makedirs(out_dir, exist_ok=True)
        inp_path = os.path.join(out_dir, "sys.inp")
        inp_content = self.generate_arpes_inp_content(**kwargs)
        with open(inp_path, "w") as f:
            f.write(inp_content)
        return inp_path
