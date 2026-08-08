# File: tensorspec/core/dft/band_service.py
"""
Band-structure orchestration: k-path construction, solver call, Fermi shift.

This sequence previously lived in the Qt suite's `calculate_bands`, so nothing
but the desktop GUI could produce a band structure. It is pure computation --
no widgets, no plotting, no file dialogs -- and returns plain arrays that any
front end can render.

The solver itself stays in `ChinookTightBindingEngine`; this module only
decides which k-points to hand it and how to shift the result.
"""
import os

import numpy as np

# Path templates the UI offers. "auto" derives the path from the lattice
# symmetry; "custom" takes explicit fractional coordinates from the caller.
# "primitive_hex_ref" samples a graphene-like Γ–K–M–Γ path in absolute Å⁻¹,
# then folds each k into the supercell BZ (educational for twisted stacks).
PATH_AUTO = "auto"
PATH_CUSTOM = "custom"
PATH_PRIMITIVE_HEX_REF = "primitive_hex_ref"
PATH_TEMPLATES = ("hexagonal", "rectangular", "square")
PATH_MODES = (PATH_AUTO, PATH_CUSTOM, PATH_PRIMITIVE_HEX_REF) + PATH_TEMPLATES

TB_SCALAR = "Simple Scalar (Isotropic)"
TB_SLATER_KOSTER = "Slater-Koster (Rigorous)"


def describe_bz_context(structure) -> dict:
    """Explain whether bands will look folded (moiré / heterostructure cell)."""
    if structure is None:
        return {
            "kind": "unknown",
            "title": "Unknown cell",
            "message": "No structure loaded.",
            "likely_folded": False,
        }
    n = len(structure)
    c = float(structure.lattice.c)
    area = float(structure.lattice.a * structure.lattice.b)
    likely_folded = (c > 12.0 and n >= 8) or n >= 16 or area > 80.0
    if likely_folded:
        return {
            "kind": "folded_supercell",
            "title": "Supercell / moiré BZ (folded)",
            "message": (
                "This cell looks like a stacked or large supercell. "
                "Default Auto path uses THIS cell's Brillouin zone, so monolayer Dirac cones "
                "appear as many folded mini-bands (correct for the moiré cell, hard to read). "
                "For lab intuition: (1) MEGNet gap for a quick scale, (2) path mode "
                "'Primitive hex reference (folded into supercell)' to walk graphene-like Γ–K–M "
                "while still solving the supercell Hamiltonian. "
                "True ARPES-style unfolding (spectral weight onto the monolayer BZ) is not "
                "implemented yet — that would be the next educational step."
            ),
            "likely_folded": True,
        }
    return {
        "kind": "standard",
        "title": "Standard / primitive-like BZ",
        "message": (
            "Auto path follows this lattice's high-symmetry lines (pymatgen). "
            "For hexagonal monolayers that is the familiar Γ–K–M path."
        ),
        "likely_folded": False,
    }


def _fold_k_into_supercell(k_cart: np.ndarray, recip_sc: np.ndarray) -> np.ndarray:
    """Map Cartesian k into the first supercell BZ by reciprocal-lattice folding."""
    inv = np.linalg.inv(recip_sc)
    frac = np.dot(k_cart, inv)
    frac = frac - np.floor(frac)
    return np.dot(frac, recip_sc)


def build_primitive_hex_reference_path(structure, points_per_segment: int = 100, a0: float = 2.46):
    """
    Dense Γ–K–M–Γ path of a reference hexagonal monolayer (Å⁻¹), with each
    point folded into the supercell's BZ for diagonalisation.

    The x-axis (`k_dist`) follows the *primitive* path length so labels stay
    familiar; the Hamiltonian is still H(k) of the supercell at the folded k.
    """
    from pymatgen.core import Lattice

    nseg = max(10, int(points_per_segment))
    ref = Lattice.hexagonal(float(a0), max(float(structure.lattice.c), 20.0))
    recip_ref = np.asarray(ref.reciprocal_lattice.matrix, dtype=float)
    recip_sc = np.asarray(structure.lattice.reciprocal_lattice.matrix, dtype=float)

    nodes_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    labels = ["Γ", "K", "M", "Γ"]
    nodes_cart = nodes_frac @ recip_ref

    prim_pts = []
    node_idx = [0]
    for i in range(len(nodes_cart) - 1):
        a, b = nodes_cart[i], nodes_cart[i + 1]
        for j in range(nseg):
            t = j / float(nseg)
            prim_pts.append((1.0 - t) * a + t * b)
        prim_pts.append(b.copy())
        node_idx.append(len(prim_pts) - 1)

    prim_pts = np.asarray(prim_pts, dtype=float)
    k_vecs = np.asarray([_fold_k_into_supercell(k, recip_sc) for k in prim_pts], dtype=float)
    k_dist = np.zeros(len(prim_pts), dtype=float)
    for i in range(1, len(prim_pts)):
        k_dist[i] = k_dist[i - 1] + float(np.linalg.norm(prim_pts[i] - prim_pts[i - 1]))

    return k_vecs, k_dist, node_idx, labels


def build_kpath(engine, path_mode: str = PATH_AUTO, custom_coords: str = "",
                custom_labels: str = "", points_per_segment: int = 100):
    """
    Produces the k-points to diagonalise along, in Cartesian reciprocal space.

    High-symmetry points arrive in fractional coordinates and are converted
    with the reciprocal lattice. Skipping that conversion silently misplaces
    every high-symmetry point on a non-cubic lattice.
    """
    if path_mode not in PATH_MODES:
        raise ValueError(f"Unknown path mode '{path_mode}'. Expected one of {PATH_MODES}.")

    structure = engine.crystal_structure
    if path_mode == PATH_PRIMITIVE_HEX_REF:
        if structure is None:
            raise ValueError("primitive_hex_ref path needs a loaded crystal structure.")
        return build_primitive_hex_reference_path(structure, points_per_segment=points_per_segment)

    if path_mode == PATH_AUTO:
        points, labels = engine.get_auto_kpath()
    elif path_mode == PATH_CUSTOM:
        points, labels = engine.get_custom_kpath(custom_coords, custom_labels)
    else:
        lattice = getattr(structure, "lattice", None) if structure is not None else None
        a = lattice.a if lattice is not None else 3.0
        b = lattice.b if lattice is not None else 3.0
        points, labels = engine.get_kpath_template(path_mode, a=a, b=b)

    if structure is not None and hasattr(structure, "lattice"):
        points = np.dot(points, structure.lattice.reciprocal_lattice.matrix)

    return engine.generate_k_path(points, labels, points_per_segment=points_per_segment)


def pack_hopping(shell_keys, amplitudes) -> dict:
    """
    Pairs hopping amplitudes with the shell names for the loaded material.

    The UI shows four anonymous t1..t4 boxes; the meaning of each comes from
    the material's own shell list, so the pairing has to happen here rather
    than being hard-coded.
    """
    return {key: float(value) for key, value in zip(shell_keys, amplitudes)}


def read_fermi_energy(reference_path: str) -> float:
    """
    Recovers the Fermi level from Quantum ESPRESSO output beside a Wannier file.

    Wannier90 eigenvalues are absolute, so without this shift the bands sit at
    the wrong energy and zero is no longer the Fermi level. Returns 0.0 when no
    output file is present, which is correct for a pure model calculation.
    """
    if not reference_path:
        return 0.0

    work_dir = os.path.dirname(reference_path)
    for name in ("nscf.out", "scf.out"):
        candidate = os.path.join(work_dir, name)
        if not os.path.exists(candidate):
            continue
        energy = 0.0
        with open(candidate, "r") as handle:
            for line in handle:
                if "the Fermi energy is" in line:
                    energy = float(line.split()[4])
        if energy != 0.0:
            return energy
    return 0.0


def calculate_bands(engine, *, path_mode: str = PATH_AUTO, custom_coords: str = "",
                    custom_labels: str = "", points_per_segment: int = 100,
                    shell_keys=(), hoppings=(), cutoffs=(1.6, 2.6, 3.1, 4.5),
                    onsite_e: float = 0.0, orbital_shifts=None,
                    use_soc: bool = False, soc_strength: float = 0.5,
                    tb_mode: str = TB_SCALAR, w90_filepath: str = None) -> dict:
    """
    Runs a 1D high-symmetry band structure and returns plain arrays.

    `engine` must already hold a crystal structure. Energies come back shifted
    so that zero is the Fermi level.
    """
    shifts = orbital_shifts or {"0": -10.0, "1": -2.0, "2": 0.0}

    k_vecs, k_dist, node_idx, labels = build_kpath(
        engine, path_mode, custom_coords, custom_labels, points_per_segment
    )

    eigenvalues, eigenvectors, orb_labels = engine.solve_bands(
        k_vecs,
        custom_hopping=pack_hopping(shell_keys, hoppings),
        onsite_e=onsite_e,
        use_soc=use_soc,
        soc_strength=soc_strength,
        w90_filepath=w90_filepath,
        cutoffs=list(cutoffs),
        tb_mode=tb_mode,
        orbital_shifts=shifts,
    )

    fermi_energy = read_fermi_energy(w90_filepath)
    eigenvalues = np.asarray(eigenvalues) - fermi_energy

    structure = engine.crystal_structure
    bz = describe_bz_context(structure)
    if path_mode == PATH_PRIMITIVE_HEX_REF:
        path_note = (
            "Primitive hex Γ–K–M–Γ in absolute Å⁻¹, each k folded into this supercell BZ. "
            "X-axis uses the monolayer path length; eigenvalues are still supercell H(k). "
            "Not true spectral-weight unfolding."
        )
        path_kind = "primitive_hex_ref_folded"
    elif bz["likely_folded"]:
        path_note = bz["message"]
        path_kind = bz["kind"]
    else:
        path_note = bz["message"]
        path_kind = bz["kind"]

    return {
        "k_vecs": np.asarray(k_vecs),
        "k_dist": np.asarray(k_dist),
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "node_idx": node_idx,
        "labels": labels,
        "orbital_labels": orb_labels,
        "fermi_energy": fermi_energy,
        "n_bands": int(eigenvalues.shape[1]) if eigenvalues.ndim == 2 else 0,
        "path_kind": path_kind,
        "path_title": bz["title"] if path_mode != PATH_PRIMITIVE_HEX_REF else "Primitive hex reference (folded)",
        "path_note": path_note,
        "likely_folded": bool(bz["likely_folded"]),
    }


def calculate_2d_mesh(
    engine,
    *,
    kx_min: float = -4.5,
    kx_max: float = 4.5,
    ky_min: float = -4.5,
    ky_max: float = 4.5,
    resolution: int = 24,
    shell_keys=(),
    hoppings=(),
    cutoffs=(1.6, 2.6, 3.1, 4.5),
    onsite_e: float = 0.0,
    orbital_shifts=None,
    use_soc: bool = False,
    soc_strength: float = 0.5,
    tb_mode: str = TB_SCALAR,
) -> dict:
    """
    Diagonalises a rectangular kx–ky mesh for ARPES matrix-element mapping.

    Option A interpolates bands onto the detector frame from this mesh. The
    resolution is intentionally capped by the caller; a shared server cannot
    afford the desktop's largest grids.
    """
    shifts = orbital_shifts or {"0": -10.0, "1": -2.0, "2": 0.0}
    res = max(4, int(resolution))
    kx_vals = np.linspace(kx_min, kx_max, res)
    ky_vals = np.linspace(ky_min, ky_max, res)
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals, indexing="ij")
    k_vecs = np.column_stack([
        kx_grid.ravel(),
        ky_grid.ravel(),
        np.zeros(res * res, dtype=float),
    ])

    eigenvalues, eigenvectors, orb_labels = engine.solve_bands(
        k_vecs,
        custom_hopping=pack_hopping(shell_keys, hoppings),
        onsite_e=onsite_e,
        use_soc=use_soc,
        soc_strength=soc_strength,
        cutoffs=list(cutoffs),
        tb_mode=tb_mode,
        orbital_shifts=shifts,
    )
    eigenvalues = np.asarray(eigenvalues)

    structure = engine.crystal_structure
    recip = None
    if structure is not None and hasattr(structure, "lattice"):
        recip = np.asarray(structure.lattice.reciprocal_lattice.matrix, dtype=float)

    return {
        "type": "band_structure",
        "is_2d": True,
        "k_vecs": np.asarray(k_vecs),
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "orbital_positions": [site.coords.tolist() for site in structure] if structure else [],
        "orbital_labels": orb_labels,
        "kx": kx_vals,
        "ky": ky_vals,
        "grid_shape": (res, res),
        "fermi_energy": 0.0,
        "tb_model": getattr(engine, "tb_model", None),
        "basis": getattr(engine, "basis", None),
        "H_dict": getattr(engine, "H_dict", None),
        "structure": structure,
        "recip_matrix": recip,
        "n_bands": int(eigenvalues.shape[1]) if eigenvalues.ndim == 2 else 0,
    }
