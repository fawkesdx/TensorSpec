"""Server-side configuration for shared deployment.

Solver paths and resource caps live here so the browser never chooses what to
execute -- only whether to queue a run. Values can be overridden with
environment variables when deploying to the LBL server.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Repository root: tensorspec/web/server/config.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SolverConfig:
    """Allowlisted Quantum ESPRESSO / Wannier90 executables and run limits."""

    pw: Path
    wannier90: Path
    pw2wannier90: Path
    mpirun: Path | None
    pseudo_dir: Path
    max_mpi_ranks: int = 8
    max_jobs_per_session: int = 1
    max_global_jobs: int = 4

    def require_exists(self) -> None:
        for label, path in (
            ("pw.x", self.pw),
            ("wannier90.x", self.wannier90),
            ("pw2wannier90.x", self.pw2wannier90),
            ("pseudopotential directory", self.pseudo_dir),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found at {path}")


def _resolve_executable(env_key: str, default_name: str, *, require: bool = True) -> Path:
    """Prefer an absolute path from the environment; otherwise search PATH."""
    override = os.environ.get(env_key, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            if require:
                raise FileNotFoundError(f"{env_key}={override} is not an executable file.")
            return path
        return path.resolve()

    found = shutil.which(default_name)
    if not found:
        if require:
            raise FileNotFoundError(
                f"'{default_name}' is not on PATH. Set {env_key} to its absolute path."
            )
        return Path(default_name)
    return Path(found).resolve()


def load_solver_config(*, require_binaries: bool = True) -> SolverConfig:
    """Build the allowlist from the environment, with sensible local defaults."""
    pseudo = Path(os.environ.get("TENSORSPEC_PSEUDO_DIR", REPO_ROOT / "Pseudo")).expanduser()
    mpirun_override = os.environ.get("TENSORSPEC_MPIRUN", "").strip()
    if mpirun_override:
        mpirun: Path | None = Path(mpirun_override).expanduser().resolve()
    else:
        found = shutil.which("mpirun")
        mpirun = Path(found).resolve() if found else None

    return SolverConfig(
        pw=_resolve_executable("TENSORSPEC_PW", "pw.x", require=require_binaries),
        wannier90=_resolve_executable("TENSORSPEC_WANNIER90", "wannier90.x", require=require_binaries),
        pw2wannier90=_resolve_executable(
            "TENSORSPEC_PW2WANNIER90", "pw2wannier90.x", require=require_binaries
        ),
        mpirun=mpirun,
        pseudo_dir=pseudo.resolve(),
        max_mpi_ranks=int(os.environ.get("TENSORSPEC_MAX_MPI_RANKS", "8")),
        max_jobs_per_session=int(os.environ.get("TENSORSPEC_MAX_JOBS_PER_SESSION", "1")),
        max_global_jobs=int(os.environ.get("TENSORSPEC_MAX_GLOBAL_JOBS", "4")),
    )
