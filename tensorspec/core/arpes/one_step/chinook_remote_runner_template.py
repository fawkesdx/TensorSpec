import os
import sys
import json
import time
import argparse
import importlib.util
import numpy as np
import concurrent.futures


import collections
import collections.abc
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

try:
    import chinook.build_lib as build_lib
    import chinook.TB_lib as tb_lib
    import chinook.klib as klib
    import chinook.orbital as olib
    import chinook.ARPES_lib as arpes_lib
except ImportError as e:
    import traceback
    print(f"FATAL: Chinook import failed!\n{traceback.format_exc()}")
    sys.exit(1)

# Load shared kmesh module (uploaded beside this script on the remote cluster).
_RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
_KMESH_PATH = os.path.join(_RUNNER_DIR, "chinook_arpes_kmesh.py")
if not os.path.isfile(_KMESH_PATH):
    print(f"FATAL: missing {_KMESH_PATH} — re-submit from GUI to upload kmesh module.", flush=True)
    sys.exit(1)
_kmesh_spec = importlib.util.spec_from_file_location("chinook_arpes_kmesh", _KMESH_PATH)
_kmesh = importlib.util.module_from_spec(_kmesh_spec)
_kmesh_spec.loader.exec_module(_kmesh)
run_chinook_arpes = _kmesh.run_chinook_arpes
run_grizzly_arpes = _kmesh.run_grizzly_arpes
apply_chinook_runtime_patches = _kmesh.apply_chinook_runtime_patches

apply_chinook_runtime_patches()

# Set in main() before ProcessPoolExecutor (inherited by fork workers).
USE_GRIZZLY = False
GRIZZLY_DEVICE = "cpu"
global_physics = None
global_B_matrix = None


def _grizzly_available():
    try:
        from grizzly import GrizzlyExperiment  # noqa: F401
        return True
    except ImportError:
        return False


def _base_arpes_args(tx_cube, ty_cube, e_cube, hv, workf, temp, polar):
    return {
        'cube': {
            'Tx': tx_cube,
            'Ty': ty_cube,
            'E': e_cube,
            'kz': 0.0
        },
        'hv': hv,
        'W': workf,
        'pol': np.array([1, 0, 0]) if polar == 'P' else np.array([0, 1, 0]),
        'T': temp,
        'resolution': {'E': 0.02, 'k': 0.01},
        'SE': ['constant', 0.05]
    }


def _normalize_full_cube(Ig, ntheta, nphi, ne):
    Ig = np.asarray(Ig, dtype=np.float32)
    if Ig.shape == (ntheta, nphi, ne):
        return Ig
    if Ig.shape == (nphi, ntheta, ne):
        print("NOTE: transposing Ig from (nphi,ntheta,ne) -> (ntheta,nphi,ne)", flush=True)
        return np.transpose(Ig, (1, 0, 2))
    if Ig.size == ntheta * nphi * ne:
        return Ig.reshape(ntheta, nphi, ne)
    raise RuntimeError(
        f"Unexpected Grizzly full-cube shape {Ig.shape}; expected "
        f"({ntheta},{nphi},{ne})"
    )


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def _cuda_empty_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_full_cube_grizzly(
    tb_model,
    thetas,
    phis,
    e_axis,
    physics,
    B_matrix,
    device,
    theta_chunk: int = 0,
):
    """Single-process full (θ,φ,E) cube on GrizzlyME + kmesh.

    Parameters
    ----------
    theta_chunk : int
        If >0, run sequential θ-blocks of this size (same process / GPU) to
        limit peak VRAM. 0 = one shot over the full θ range.
    """
    ntheta, nphi, ne = len(thetas), len(phis), len(e_axis)
    t_all = time.perf_counter()

    if theta_chunk and theta_chunk > 0 and theta_chunk < ntheta:
        print(
            f"FULL layout (θ-chunk={theta_chunk}): GrizzlyME+kmesh "
            f"({ntheta} x {nphi} x {ne}) device={device}",
            flush=True,
        )
        cube = np.zeros((ntheta, nphi, ne), dtype=np.float32)
        n_blocks = (ntheta + theta_chunk - 1) // theta_chunk
        for i_block, i0 in enumerate(range(0, ntheta, theta_chunk)):
            i1 = min(i0 + theta_chunk, ntheta)
            th = thetas[i0:i1]
            k_bounds = {
                "X": [float(th[0]), float(th[-1]), len(th)],
                "Y": [float(phis[0]), float(phis[-1]), nphi],
                "E": [float(e_axis[0]), float(e_axis[-1]), ne],
            }
            print(
                f"  θ-chunk {i_block + 1}/{n_blocks}: "
                f"idx[{i0}:{i1}] θ=[{th[0]:.3f},{th[-1]:.3f}] ({len(th)} pts)",
                flush=True,
            )
            t_blk = time.perf_counter()
            try:
                Ig = run_grizzly_arpes(
                    tb_model,
                    k_bounds,
                    physics,
                    B_matrix,
                    fermi_shift=0.0,
                    device=device,
                    profile_stages=True,
                )
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    print(
                        "CUDA OOM during θ-chunked GrizzlyME "
                        "(caller may retry smaller chunk or fall back).",
                        flush=True,
                    )
                raise
            cube[i0:i1, :, :] = _normalize_full_cube(Ig, len(th), nphi, ne)
            _cuda_empty_cache()
            print(
                f"    chunk wall: {time.perf_counter() - t_blk:.2f}s",
                flush=True,
            )
        print(f"  full-cube wall: {time.perf_counter() - t_all:.2f}s", flush=True)
        return cube

    k_bounds = {
        "X": [float(thetas[0]), float(thetas[-1]), ntheta],
        "Y": [float(phis[0]), float(phis[-1]), nphi],
        "E": [float(e_axis[0]), float(e_axis[-1]), ne],
    }
    print(
        f"FULL layout: GrizzlyME+kmesh cube ({ntheta} x {nphi} x {ne}) device={device}",
        flush=True,
    )
    try:
        Ig = run_grizzly_arpes(
            tb_model,
            k_bounds,
            physics,
            B_matrix,
            fermi_shift=0.0,
            device=device,
            profile_stages=True,
        )
    except RuntimeError as exc:
        if _is_oom_error(exc):
            print(
                "CUDA OOM during full-cube GrizzlyME (will allow caller to fall back).",
                flush=True,
            )
        raise

    Ig = _normalize_full_cube(Ig, ntheta, nphi, ne)
    print(f"  full-cube wall: {time.perf_counter() - t_all:.2f}s", flush=True)
    return Ig


def run_single_theta_slice(theta_idx, theta_val, phis, e_axis, sarpes_str, spin_axis, spin_comp):
    global global_tb_model, global_physics, global_B_matrix, USE_GRIZZLY, GRIZZLY_DEVICE
    ne = len(e_axis)
    nphi = len(phis)
    slice_intensity = np.zeros((nphi, ne), dtype=np.float32)

    use_spin = sarpes_str.lower() == "true"
    if use_spin:
        # SARPES: legacy chinook cube path (kmesh spin handling not yet unified).
        arpes_args = _base_arpes_args(
            (theta_val, theta_val, 1),
            (phis[0], phis[-1], nphi),
            (e_axis[0], e_axis[-1], ne),
            global_physics["hv"],
            global_physics["work_function"],
            global_physics["temperature"],
            "P",
        )
        axis_vec = np.array([0, 0, 1])
        if spin_axis == "X":
            axis_vec = np.array([1, 0, 0])
        elif spin_axis == "Y":
            axis_vec = np.array([0, 1, 0])
        arpes_args["spin"] = [spin_comp, axis_vec]
        experiment = arpes_lib.experiment(global_tb_model, arpes_args)
        if experiment is None:
            return theta_idx, slice_intensity
        _I, Ig = experiment.spectral()
        Ig = np.asarray(Ig)
        if Ig.ndim == 3:
            if Ig.shape[0] == 1:
                slice_intensity = Ig[0, :, :]
            elif Ig.shape[1] == 1:
                slice_intensity = Ig[:, 0, :]
        elif Ig.ndim == 2:
            slice_intensity = Ig
        return theta_idx, np.asarray(slice_intensity, dtype=np.float32)

    k_bounds = {
        "X": [float(theta_val), float(theta_val), 1],
        "Y": [float(phis[0]), float(phis[-1]), nphi],
        "E": [float(e_axis[0]), float(e_axis[-1]), ne],
    }
    try:
        if USE_GRIZZLY:
            intensity_3d = run_grizzly_arpes(
                global_tb_model,
                k_bounds,
                global_physics,
                global_B_matrix,
                fermi_shift=0.0,
                device=GRIZZLY_DEVICE,
            )
        else:
            intensity_3d = run_chinook_arpes(
                global_tb_model,
                k_bounds,
                global_physics,
                global_B_matrix,
                fermi_shift=0.0,
            )
        slice_intensity = np.asarray(intensity_3d[0, :, :], dtype=np.float32)
    except Exception as exc:
        print(f"WARNING: theta[{theta_idx}]={theta_val} failed: {exc}", flush=True)

    return theta_idx, slice_intensity


def main():
    global USE_GRIZZLY, GRIZZLY_DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--tb_file", type=str, required=True)
    parser.add_argument("--theta_min", type=float, required=True)
    parser.add_argument("--theta_max", type=float, required=True)
    parser.add_argument("--ntheta", type=int, required=True)
    parser.add_argument("--phi_min", type=float, required=True)
    parser.add_argument("--phi_max", type=float, required=True)
    parser.add_argument("--nphi", type=int, required=True)
    parser.add_argument("--e_min", type=float, default=-2.0)
    parser.add_argument("--e_max", type=float, default=0.5)
    parser.add_argument("--ne", type=int, default=100)
    parser.add_argument("--hv", type=float, default=90.0)
    parser.add_argument("--workf", type=float, default=4.5)
    parser.add_argument("--v0", type=float, default=15.0)
    parser.add_argument("--temp", type=float, default=10.0)
    parser.add_argument("--polar", type=str, default="P")
    parser.add_argument("--cores", type=int, default=40)
    parser.add_argument("--sarpes", type=str, default="False")
    parser.add_argument("--spin_axis", type=str, default="Z")
    parser.add_argument("--spin_comp", type=int, default=1)
    parser.add_argument(
        "--engine",
        type=str,
        default="auto",
        choices=("auto", "grizzly", "chinook"),
        help="ME engine: auto=GrizzlyME if installed (spinless), else chinook",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="GrizzlyME device (ignored for chinook). Slice+CUDA caps workers at 2.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="auto",
        choices=("auto", "slices", "full"),
        help="auto: full when Grizzly+CUDA+spinless; else θ-slices ProcessPool",
    )
    parser.add_argument(
        "--theta_chunk",
        type=int,
        default=0,
        help="FULL layout only: sequential θ-block size (0=one-shot). "
        "On CUDA OOM, runner also auto-retries with shrinking chunks.",
    )
    parser.add_argument("--out_file", type=str, default="chinook_arpes_cube.npz")
    parser.add_argument(
        "--e_fermi",
        type=float,
        default=None,
        help="Override Fermi energy (eV). Default: value stored in tb_file (e_fermi).",
    )
    parser.add_argument(
        "--physics_file",
        type=str,
        default="arpes_physics.json",
        help="JSON with beamline/manipulator/hkl settings (same as local GUI).",
    )
    args = parser.parse_args()

    want_grizzly = args.engine in ("auto", "grizzly")
    have_grizzly = _grizzly_available()
    sarpes_on = args.sarpes.lower() == "true"

    if args.engine == "grizzly" and not have_grizzly:
        print("FATAL: --engine grizzly but grizzlyme is not installed in this Python.", flush=True)
        sys.exit(1)
    if args.engine == "grizzly" and sarpes_on:
        print("FATAL: GrizzlyME v0.1 is spinless; disable SARPES or use --engine chinook.", flush=True)
        sys.exit(1)

    USE_GRIZZLY = bool(want_grizzly and have_grizzly and not sarpes_on)
    GRIZZLY_DEVICE = args.device
    if USE_GRIZZLY and GRIZZLY_DEVICE == "auto":
        try:
            import torch
            GRIZZLY_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            GRIZZLY_DEVICE = "cpu"

    if args.layout == "auto":
        layout = "full" if (USE_GRIZZLY and GRIZZLY_DEVICE == "cuda") else "slices"
    else:
        layout = args.layout

    if layout == "full" and not USE_GRIZZLY:
        print("FATAL: --layout full requires GrizzlyME (spinless). Use --engine grizzly/auto.", flush=True)
        sys.exit(1)
    if layout == "full" and sarpes_on:
        print("NOTE: SARPES + full layout unsupported; falling back to slices/chinook.", flush=True)
        layout = "slices"

    engine_name = "GrizzlyME" if USE_GRIZZLY else "chinook"
    print(
        f"Starting ARPES map on cluster (engine={engine_name}"
        + (f", device={GRIZZLY_DEVICE}" if USE_GRIZZLY else "")
        + f", layout={layout})...",
        flush=True,
    )
    if want_grizzly and not have_grizzly:
        print("NOTE: grizzlyme not installed — falling back to chinook.", flush=True)
    if want_grizzly and sarpes_on:
        print("NOTE: SARPES requested — GrizzlyME skipped (spinless v0.1); using chinook.", flush=True)

    # --- LOAD TB DATA ---
    data = np.load(args.tb_file, allow_pickle=True)
    indices = data['indices']
    values = data['values']
    basis_list = data['basis_list']
    a_mat = data['a_mat'].tolist() if 'a_mat' in data else np.eye(3).tolist()

    if 'b_matrix' in data:
        b_matrix = np.asarray(data['b_matrix'], dtype=float)
    else:
        A = np.asarray(a_mat, dtype=float)
        b_matrix = 2 * np.pi * np.linalg.inv(A).T
        print("NOTE: b_matrix missing in tb_file; derived from a_mat.", flush=True)

    physics_path = args.physics_file
    if not os.path.isabs(physics_path):
        physics_path = os.path.join(_RUNNER_DIR, physics_path)
    if not os.path.isfile(physics_path):
        print(f"FATAL: physics file missing: {physics_path}", flush=True)
        sys.exit(1)
    with open(physics_path, "r") as f:
        physics = json.load(f)
    physics.setdefault("hkl", [0, 0, 1])
    physics["hkl"] = tuple(int(x) for x in physics["hkl"])
    # CLI overrides for beam energy / surface params (match GUI spinboxes on submit).
    physics["hv"] = float(args.hv)
    physics["work_function"] = float(args.workf)
    physics["inner_potential"] = float(args.v0)
    physics["temperature"] = float(args.temp)
    global global_physics, global_B_matrix
    global_physics = physics
    global_B_matrix = b_matrix
    print(
        f"Loaded physics: V0={physics.get('inner_potential')} eV, "
        f"hkl={physics.get('hkl')}, pol={physics.get('polarization')}, "
        f"ME={physics.get('matrix_element_mode')}",
        flush=True,
    )

    if np.issubdtype(indices.dtype, np.integer):
        print(
            "WARNING: tb_file indices are integer-typed. Cartesian hopping R may have "
            "been truncated on upload — intensity can be all zeros at finite angle. "
            "Re-submit from GUI after float64 upload fix.",
            flush=True,
        )

    # Orbital indices are ints; R (cols 2:5) must remain float (Cartesian Å).
    explicit_hopping = []
    for i in range(len(indices)):
        explicit_hopping.append([
            int(indices[i, 0]),
            int(indices[i, 1]),
            float(indices[i, 2]),
            float(indices[i, 3]),
            float(indices[i, 4]),
            complex(values[i]),
        ])

    e_fermi = float(args.e_fermi) if args.e_fermi is not None else float(
        data['e_fermi'] if 'e_fermi' in data else 0.0
    )
    print(f"e_fermi for eigenvalue shift: {e_fermi} eV", flush=True)

    tb_dict = {
        'type': 'list',
        'list': explicit_hopping,
        'H': explicit_hopping,
        'a': a_mat,
        'spin': {'bool': False, 'soc': False}
    }

    bulk_basis = []
    for i, b in enumerate(basis_list):
        orb = olib.orbital(i, i, str(b['label']), b['pos'], int(b.get('Z', 1)), spin=b.get('spin', 1.0))
        bulk_basis.append(orb)

    global global_tb_model
    global_tb_model = tb_lib.TB_model(bulk_basis, tb_dict, klib.kpath(np.array([[0,0,0]])))
    print("Successfully reconstructed TB Model with", len(explicit_hopping), "hoppings!")

    # Match local chinook_wrapper: shift eigenvalues so EF sits at 0.
    if abs(e_fermi) > 1e-12:
        _orig_solve_h = global_tb_model.solve_H

        def _solve_h_ef(Eonly=False, _orig=_orig_solve_h, _ef=e_fermi):
            Eband, Evec = _orig(Eonly=Eonly)
            Eband = np.asarray(Eband) - _ef
            return Eband, Evec

        global_tb_model.solve_H = _solve_h_ef
        print(f"Patched TB.solve_H to subtract e_fermi={e_fermi} eV", flush=True)

    e_kin = max(args.hv - args.workf, 0.1)
    k_radius = 0.512316 * np.sqrt(e_kin)

    thetas = np.linspace(args.theta_min, args.theta_max, args.ntheta)
    phis = np.linspace(args.phi_min, args.phi_max, args.nphi)
    e_axis = np.linspace(args.e_min, args.e_max, args.ne)

    # Match local chinook_wrapper: degenerate range → exactly 1 sample.
    if args.theta_min == args.theta_max:
        thetas = np.array([args.theta_min], dtype=float)
        print(f"NOTE: theta_min==theta_max ({args.theta_min}) → ntheta  {args.ntheta} -> 1", flush=True)
    if args.phi_min == args.phi_max:
        phis = np.array([args.phi_min], dtype=float)
        print(f"NOTE: phi_min==phi_max ({args.phi_min}) → nphi {args.nphi} -> 1", flush=True)
    if args.e_min == args.e_max:
        e_axis = np.array([args.e_min], dtype=float)
        print(f"NOTE: e_min==e_max ({args.e_min}) → ne {args.ne} -> 1", flush=True)

    # Keep args.ntheta/nphi/ne consistent with collapsed axes for cube alloc / progress.
    args.ntheta = len(thetas)
    args.nphi = len(phis)
    args.ne = len(e_axis)

    num_hoppings = len(explicit_hopping)
    del tb_dict
    del explicit_hopping
    del indices
    del values
    del data
    import gc
    gc.collect()

    class CudaOOMError(RuntimeError):
        pass

    def _run_slices_path():
        cube_out = np.zeros((args.ntheta, args.nphi, args.ne), dtype=np.float32)

        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        gb_per_worker = max(0.5, (num_hoppings / 1000000.0) * 1.5)
        max_safe_workers = max(1, int((available_gb * 0.9) / gb_per_worker))

        safe_cores = min(args.cores, max_safe_workers)
        if USE_GRIZZLY and GRIZZLY_DEVICE == "cuda" and safe_cores > 2:
            print(
                f"NOTE: GrizzlyME CUDA slices — throttling workers {safe_cores} -> 2 "
                f"(avoid multi-process GPU contention).",
                flush=True,
            )
            safe_cores = 2
        elif safe_cores < args.cores:
            print(f"WARNING: MEMORY LIMIT DETECTED: {available_gb:.1f} GB available.")
            print(f"WARNING: Each worker needs ~{gb_per_worker:.1f} GB due to {num_hoppings} hoppings.")
            print(f"WARNING: Automatically throttling from {args.cores} down to {safe_cores} cores to prevent OOM!")

        with concurrent.futures.ProcessPoolExecutor(max_workers=safe_cores) as executor:
            futures = {
                executor.submit(
                    run_single_theta_slice,
                    itheta,
                    theta_val,
                    phis,
                    e_axis,
                    args.sarpes,
                    args.spin_axis,
                    args.spin_comp,
                ): itheta for itheta, theta_val in enumerate(thetas)
            }

            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                itheta, slice_res = fut.result()
                cube_out[itheta, :, :] = slice_res
                completed += 1
                if completed % max(1, args.ntheta // 10) == 0 or completed == args.ntheta:
                    pct = int(completed / args.ntheta * 100)
                    print(f"Progress: {completed}/{args.ntheta} angle slices ({pct}%) complete...", flush=True)
        return cube_out

    used_layout = layout
    used_theta_chunk = int(args.theta_chunk) if args.theta_chunk else 0
    if layout == "full":
        chunk_try = used_theta_chunk
        chunk_schedule = [chunk_try]
        # Auto OOM ladder: one-shot → half θ → 20 → 10 → 5 → 1, then slices.
        for c in (max(1, args.ntheta // 2), 20, 10, 5, 1):
            if c not in chunk_schedule and c < args.ntheta:
                chunk_schedule.append(c)
        cube = None
        last_exc = None
        for chunk_try in chunk_schedule:
            try:
                cube = run_full_cube_grizzly(
                    global_tb_model,
                    thetas,
                    phis,
                    e_axis,
                    global_physics,
                    global_B_matrix,
                    GRIZZLY_DEVICE,
                    theta_chunk=chunk_try,
                )
                used_theta_chunk = chunk_try
                last_exc = None
                break
            except RuntimeError as exc:
                last_exc = exc
                if not _is_oom_error(exc):
                    print(f"FATAL: full-cube failed: {exc}", flush=True)
                    sys.exit(1)
                print(
                    f"WARNING: CUDA OOM with theta_chunk={chunk_try}. "
                    "Clearing GPU cache; trying smaller chunk.",
                    flush=True,
                )
                _cuda_empty_cache()
        if cube is None:
            print(
                "FATAL: CUDA OOM on all full-cube chunk sizes. "
                "Refusing slices fallback for Grizzly CUDA (use smaller --theta_chunk).",
                flush=True,
            )
            if last_exc is not None:
                print(f"Last error: {last_exc}", flush=True)
            sys.exit(1)
    else:
        cube = _run_slices_path()

    print(f"Saving ARPES intensity cube to {args.out_file}...", flush=True)
    np.savez_compressed(
        args.out_file,
        cube=cube,
        energy=e_axis,
        theta=thetas,
        phi=phis,
        engine=np.array(engine_name),
        layout=np.array(used_layout),
        device=np.array(GRIZZLY_DEVICE if USE_GRIZZLY else "n/a"),
        theta_chunk=np.array(used_theta_chunk if used_layout == "full" else 0),
    )
    print(
        f"Remote ARPES calculation completed successfully "
        f"(engine={engine_name}, layout={used_layout}"
        + (f", theta_chunk={used_theta_chunk}" if used_layout == "full" else "")
        + ")!",
        flush=True,
    )


if __name__ == "__main__":
    main()
