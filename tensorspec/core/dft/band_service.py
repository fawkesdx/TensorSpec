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
# "unfold_hex" does the same and attaches Popescu–Zunger-style spectral weights.
PATH_AUTO = "auto"
PATH_CUSTOM = "custom"
PATH_PRIMITIVE_HEX_REF = "primitive_hex_ref"
PATH_UNFOLD_HEX = "unfold_hex"
PATH_TEMPLATES = ("hexagonal", "rectangular", "square")
PATH_MODES = (PATH_AUTO, PATH_CUSTOM, PATH_PRIMITIVE_HEX_REF, PATH_UNFOLD_HEX) + PATH_TEMPLATES

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
                "Default Auto path uses THIS cell's Brillouin zone (folded mini-bands). "
                "For lab intuition: MEGNet gap for scale; "
                "'Primitive hex reference' for Γ–K–M labels; "
                "'Unfold hex (spectral weight)' for ARPES-like intensity on the monolayer path "
                "(TB Popescu–Zunger weights onto a reference hex BZ)."
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


def build_primitive_hex_reference_path(
    structure,
    points_per_segment: int = 100,
    a0: float = 2.46,
    *,
    return_primitive: bool = False,
):
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

    if return_primitive:
        return k_vecs, k_dist, node_idx, labels, prim_pts
    return k_vecs, k_dist, node_idx, labels


def orbital_basis_positions(engine, structure, *, use_soc: bool = False) -> np.ndarray:
    """Cartesian Å positions for each TB basis orbital (atom-major W90 order)."""
    positions = []
    for site in structure:
        n_orbs = len(engine._get_orbital_basis(site.species_string))
        coord = np.asarray(site.coords, dtype=float)
        for _ in range(n_orbs):
            positions.append(coord)
    if use_soc:
        # Chinook spin-doubles the basis; both spins share the same atomic sites.
        positions = positions + positions
    return np.asarray(positions, dtype=float)


def spectral_weight_unfold(
    eigenvectors: np.ndarray,
    orbital_positions: np.ndarray,
    g_vecs: np.ndarray,
) -> np.ndarray:
    """
    Popescu–Zunger-style TB spectral weights onto primitive-path k-points.

    For each supercell eigenstate |ψ_n(k_sc)> with coefficients C_μn and
    unfolding vector G = k_prim − k_sc:

        W_n(k_prim) = |Σ_μ C_μn exp(−i G · r_μ)|²

    Assumes an orthogonal localized basis (standard TB approximation).
    Returns shape (nk, nband) with values clipped to [0, 1].
    """
    evecs = np.asarray(eigenvectors)
    if evecs.ndim != 3:
        raise ValueError(f"eigenvectors must be (nk, norb, nband); got {evecs.shape}")
    nk, norb, _nband = evecs.shape
    pos = np.asarray(orbital_positions, dtype=float)
    g = np.asarray(g_vecs, dtype=float)
    if pos.shape[0] != norb:
        raise ValueError(
            f"orbital_positions length {pos.shape[0]} != eigenvector basis {norb}"
        )
    if g.shape != (nk, 3):
        raise ValueError(f"g_vecs must be (nk, 3); got {g.shape}")

    # phase[k, μ] = exp(−i G_k · r_μ)
    phase = np.exp(-1j * (g @ pos.T))  # (nk, norb)
    # amp[k, n] = Σ_μ C[k,μ,n] * phase[k,μ]
    amp = np.einsum("km,kmn->kn", phase, evecs)
    weights = np.abs(amp) ** 2
    return np.clip(weights.real, 0.0, 1.0)


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
    if path_mode in (PATH_PRIMITIVE_HEX_REF, PATH_UNFOLD_HEX):
        if structure is None:
            raise ValueError(f"{path_mode} path needs a loaded crystal structure.")
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

    path_mode ``unfold_hex`` also returns Popescu–Zunger spectral weights
    onto a reference hexagonal Γ–K–M path (ARPES-like intensity).
    """
    shifts = orbital_shifts or {"0": -10.0, "1": -2.0, "2": 0.0}
    structure = engine.crystal_structure
    do_unfold = path_mode == PATH_UNFOLD_HEX
    prim_pts = None

    if do_unfold:
        if structure is None:
            raise ValueError("unfold_hex needs a loaded crystal structure.")
        k_vecs, k_dist, node_idx, labels, prim_pts = build_primitive_hex_reference_path(
            structure, points_per_segment=points_per_segment, return_primitive=True
        )
    else:
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
    eigenvectors = np.asarray(eigenvectors)

    weights = None
    if do_unfold and prim_pts is not None:
        g_vecs = np.asarray(prim_pts, dtype=float) - np.asarray(k_vecs, dtype=float)
        orb_pos = orbital_basis_positions(engine, structure, use_soc=use_soc)
        norb = eigenvectors.shape[1]
        if orb_pos.shape[0] > norb:
            orb_pos = orb_pos[:norb]
        elif orb_pos.shape[0] < norb:
            pad = np.repeat(orb_pos[-1:], norb - orb_pos.shape[0], axis=0)
            orb_pos = np.vstack([orb_pos, pad])
        weights = spectral_weight_unfold(eigenvectors, orb_pos, g_vecs)

    bz = describe_bz_context(structure)
    if path_mode == PATH_UNFOLD_HEX:
        path_kind = "unfold_hex"
        path_title = "Unfolded hex path (spectral weight)"
        path_note = (
            "Supercell H(k) at folded k; intensity = |Σ_μ C_μ exp(−iG·r_μ)|² "
            "with G = k_prim − k_folded (TB Popescu–Zunger). Bright = strong monolayer "
            "character. Approximate for twisted/orthorhombic stacks (reference a₀=2.46 Å)."
        )
    elif path_mode == PATH_PRIMITIVE_HEX_REF:
        path_kind = "primitive_hex_ref_folded"
        path_title = "Primitive hex reference (folded)"
        path_note = (
            "Primitive hex Γ–K–M–Γ in absolute Å⁻¹, each k folded into this supercell BZ. "
            "X-axis uses the monolayer path length; eigenvalues are still supercell H(k). "
            "Use Unfold hex for spectral weights."
        )
    elif bz["likely_folded"]:
        path_kind = bz["kind"]
        path_title = bz["title"]
        path_note = bz["message"]
    else:
        path_kind = bz["kind"]
        path_title = bz["title"]
        path_note = bz["message"]

    out = {
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
        "path_title": path_title,
        "path_note": path_note,
        "likely_folded": bool(bz["likely_folded"]),
    }
    if weights is not None:
        out["weights"] = weights
    return out


# Orbital character tags used in Chinook / TensorSpec labels: `{el}_{orb}` or
# `{el}_{orb}_up` / `_dn` under SOC.
_SHELL_SUFFIXES = {
    "s": ("_s",),
    "p": ("_pz", "_px", "_py"),
    "d": ("_dz2", "_dxz", "_dyz", "_dx2-y2", "_dxy"),
}


def _strip_spin_suffix(label: str) -> str:
    if label.endswith("_up") or label.endswith("_dn"):
        return label[:-3]
    return label


def resolve_fat_indices(orbital_labels, fat_target: str) -> list[int]:
    """
    Map a fat-band target string to basis-orbital indices.

    Accepted forms: ``none``, ``shell:s|p|d``, ``element:C``, ``orbital:C_pz``,
    or a bare unique label ``C_pz``.
    """
    labels = [str(x) for x in (orbital_labels or [])]
    raw = (fat_target or "none").strip()
    if not raw or raw.lower() in ("none", "none (standard lines)"):
        return []

    target = raw
    if target.startswith("shell:"):
        shell = target.split(":", 1)[1].strip().lower()
        suffixes = _SHELL_SUFFIXES.get(shell)
        if not suffixes:
            raise ValueError(f"Unknown shell '{shell}'. Use s, p, or d.")
        idxs = []
        for i, lab in enumerate(labels):
            base = _strip_spin_suffix(lab)
            if any(base.endswith(suf) for suf in suffixes):
                idxs.append(i)
        return idxs

    if target.startswith("element:"):
        el = target.split(":", 1)[1].strip()
        if not el:
            raise ValueError("element: target needs a symbol, e.g. element:C")
        prefix = f"{el}_"
        return [i for i, lab in enumerate(labels) if lab.startswith(prefix)]

    if target.startswith("orbital:"):
        want = target.split(":", 1)[1].strip()
    else:
        want = target.strip()

    exact = [i for i, lab in enumerate(labels) if lab == want]
    if exact:
        return exact
    # Allow matching without spin tag when user picked base orbital
    soft = [i for i, lab in enumerate(labels) if _strip_spin_suffix(lab) == want]
    if soft:
        return soft
    raise ValueError(f"No orbital labelled '{want}' in the cached band structure.")


def fat_band_weights(eigenvectors: np.ndarray, indices) -> np.ndarray:
    """
    Sum |C|² over selected orbital indices → shape (nk, nband), clipped to [0, 1].
    """
    evecs = np.asarray(eigenvectors)
    if evecs.ndim != 3:
        raise ValueError(f"eigenvectors must be (nk, norb, nband); got {evecs.shape}")
    idxs = [int(i) for i in indices]
    nk, norb, nband = evecs.shape
    if not idxs:
        return np.zeros((nk, nband), dtype=float)
    for i in idxs:
        if i < 0 or i >= norb:
            raise ValueError(f"Orbital index {i} out of range for basis size {norb}")
    probs = np.abs(evecs) ** 2
    weights = np.sum(probs[:, idxs, :], axis=1)
    return np.clip(weights.real, 0.0, 1.0)


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


def isoenergy_density(eigenvalues, energy, smear, grid_shape):
    """Gaussian density of states on a 2D k-mesh at fixed energy.

    I = Σ_n exp(−(E_n − E)² / (2 σ²)) per k-point, reshaped to ``grid_shape``.
    ``eigenvalues`` is (nk, nb) with nk = prod(grid_shape), matching
    ``calculate_2d_mesh`` ravel order (indexing='ij').
    """
    ev = np.asarray(eigenvalues, dtype=float)
    sigma = float(smear)
    gauss = np.exp(-((ev - float(energy)) ** 2) / (2.0 * sigma * sigma))
    intensity = np.sum(gauss, axis=-1)
    return intensity.reshape(tuple(grid_shape))
