# File: tensorspec/core/dft/qe_pipeline.py
"""
Quantum ESPRESSO input packaging and command construction.

Generates the SCF / NSCF / Wannier90 inputs into a run directory and builds the
argument lists a runner would execute. No subprocess calls live here: the web
job queue and any HPC script share the same command list so they cannot drift.

Security contract
-----------------
* Run names are restricted to a safe character set before they become directory
  names.
* Executable paths are never taken from the caller; the caller only supplies
  validated numeric parameters and a resolved SolverPaths object from server
  configuration.
* Commands are returned as argv lists suitable for ``subprocess`` with
  ``shell=False``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Structure

from tensorspec.core.dft.qe_generator import QEInputGenerator

# Run directories are named from user input; keep them boring filesystem tokens.
SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

INPUT_FILES = (
    "scf.in",
    "nscf.in",
    "wannier90.win",
    "pw2wan.in",
    "run_pipeline.sh",
)


@dataclass(frozen=True)
class SolverPaths:
    """Resolved absolute paths to allowlisted solvers."""

    pw: Path
    wannier90: Path
    pw2wannier90: Path
    mpirun: Path | None = None


@dataclass(frozen=True)
class PipelineParams:
    ecutwfc: float = 60.0
    ecutrho: float | None = None
    nbnd: int = 12
    kx: int = 6
    ky: int = 6
    kz: int = 6
    use_soc: bool = False
    mlwf_mode: bool = False
    use_mpi: bool = True
    mpi_ranks: int = 4
    # 2D / vacuum-slab QE: force kz=1 and assume_isolated='2D' in inputs
    slab_mode: bool = False

    @property
    def rho(self) -> float:
        return self.ecutrho if self.ecutrho is not None else 4.0 * self.ecutwfc

    @property
    def kmesh(self) -> tuple[int, int, int]:
        kz = 1 if self.slab_mode else self.kz
        return (self.kx, self.ky, int(kz))


def sanitize_run_name(name: str) -> str:
    """Reject path traversal and empty names before they touch the filesystem."""
    cleaned = (name or "").strip()
    if not SAFE_RUN_NAME.match(cleaned):
        raise ValueError(
            "Run name must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-' (max 64 characters)."
        )
    if ".." in cleaned or cleaned.startswith("."):
        raise ValueError("Run name cannot be a relative path.")
    return cleaned


def resolve_run_dir(session_root: Path, run_name: str) -> Path:
    """Place every run under ``{session}/qe_runs/{name}`` and nowhere else."""
    safe = sanitize_run_name(run_name)
    root = Path(session_root).resolve()
    run_dir = (root / "qe_runs" / safe).resolve()
    if not str(run_dir).startswith(str(root)):
        raise ValueError("Run directory escaped the session workspace.")
    return run_dir


def generate_inputs(
    structure: Structure,
    run_dir: Path,
    params: PipelineParams,
    *,
    pseudo_dir: Path,
    relative_outdir: bool = False,
) -> list[str]:
    """
    Write the full QE + Wannier90 input set into ``run_dir``.

    ``relative_outdir`` keeps ``outdir = './out/'`` so a downloadable bundle
    remains portable to an HPC filesystem.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    generator = QEInputGenerator(structure, pseudo_dir=pseudo_dir)
    generator.write_scf_input(
        str(run_dir),
        ecutwfc=params.ecutwfc,
        ecutrho=params.rho,
        kmesh=params.kmesh,
        use_soc=params.use_soc,
        relative_outdir=relative_outdir,
        slab_mode=params.slab_mode,
    )
    generator.write_nscf_input(
        str(run_dir),
        ecutwfc=params.ecutwfc,
        ecutrho=params.rho,
        kmesh=params.kmesh,
        nbnd=params.nbnd,
        use_soc=params.use_soc,
        relative_outdir=relative_outdir,
        slab_mode=params.slab_mode,
    )
    generator.write_wannier90_input(
        str(run_dir),
        kmesh=params.kmesh,
        num_wann=params.nbnd,
        use_soc=params.use_soc,
        mlwf_mode=params.mlwf_mode,
    )
    generator.write_pw2wan_input(str(run_dir), relative_outdir=relative_outdir)

    written = [name for name in ("scf.in", "nscf.in", "wannier90.win", "pw2wan.in")
               if (run_dir / name).is_file()]
    return written


def build_pipeline_commands(
    solvers: SolverPaths,
    params: PipelineParams,
    *,
    max_mpi_ranks: int,
) -> list[list[str]]:
    """
    Construct the argv lists for one full pipeline, with no shell involved.

    MPI wrapping is applied only to the QE executables; Wannier90 is serial.
    Rank counts are clipped to the server cap rather than trusted from the UI.
    """
    ranks = max(1, min(int(params.mpi_ranks), int(max_mpi_ranks)))
    use_mpi = bool(params.use_mpi) and solvers.mpirun is not None and ranks > 1

    def wrap(executable: Path, *args: str) -> list[str]:
        cmd = [str(executable), *args]
        if use_mpi:
            return [str(solvers.mpirun), "-np", str(ranks), *cmd]
        return cmd

    return [
        wrap(solvers.pw, "-ndiag", "1", "-in", "scf.in"),
        wrap(solvers.pw, "-ndiag", "1", "-in", "nscf.in"),
        [str(solvers.wannier90), "-pp", "wannier90"],
        wrap(solvers.pw2wannier90, "-in", "pw2wan.in"),
        [str(solvers.wannier90), "wannier90"],
    ]


def write_hpc_script(
    run_dir: Path,
    params: PipelineParams,
    *,
    max_mpi_ranks: int,
) -> Path:
    """
    Write a portable bash script that uses bare solver names for HPC clusters.

    This file is for humans to download and edit; the server never executes it.
    """
    ranks = max(1, min(int(params.mpi_ranks), int(max_mpi_ranks)))
    mpi_prefix = f"mpirun -np {ranks} " if params.use_mpi and ranks > 1 else ""

    script = f"""#!/bin/bash
# TensorSpec QE pipeline — edit and submit on your own HPC allocation.
# Generated for portable use: solvers are bare names expected on PATH.
set -euo pipefail
cd "$(dirname "$0")"

{mpi_prefix}pw.x -ndiag 1 -in scf.in | tee scf.out
{mpi_prefix}pw.x -ndiag 1 -in nscf.in | tee nscf.out
wannier90.x -pp wannier90
{mpi_prefix}pw2wannier90.x -in pw2wan.in | tee pw2wan.out
wannier90.x wannier90
"""
    path = Path(run_dir) / "run_pipeline.sh"
    path.write_text(script)
    path.chmod(0o755)
    return path
