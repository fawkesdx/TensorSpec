"""Progress + ETA helpers for running SPR-KKR jobs. Zero Qt.

Reuses ``outputs.parse_scf_log`` / ``ScfStatus`` for SCF progress rather than
re-parsing the log with a second regex set.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .outputs import ScfStatus, parse_scf_log

PathLike = Union[str, Path]

_HASH_LINE_RE = re.compile(r"^#+\s*$")


def arpes_rows_done(spc_path: PathLike) -> int:
    """Count completed data rows in a (possibly still-being-written) .spc file.

    0 if the file does not exist yet. Robust to a partial last line (still
    being flushed by the running binary) by requiring >= 8 parseable float
    columns per row.
    """
    p = Path(spc_path)
    if not p.exists():
        return 0
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except OSError:
        return 0

    in_data = False
    count = 0
    for line in lines:
        stripped = line.strip()
        if not in_data:
            if _HASH_LINE_RE.match(stripped):
                in_data = True
            continue
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        try:
            for tok in parts[:8]:
                float(tok.replace("D", "E").replace("d", "e"))
        except ValueError:
            continue
        count += 1
    return count


def arpes_fraction_done(spc_path: PathLike, params_or_expected_rows) -> float:
    """Fraction complete in [0, 1]. Second arg is an ArpesParams (uses
    ``.n_points``) or a plain int row count."""
    expected = getattr(params_or_expected_rows, "n_points", None)
    if expected is None:
        expected = int(params_or_expected_rows)
    if expected <= 0:
        return 0.0
    done = arpes_rows_done(spc_path)
    return min(1.0, done / expected)


def scf_progress(log_path: PathLike) -> ScfStatus:
    """``parse_scf_log`` on a possibly-not-yet-existing log file."""
    p = Path(log_path)
    if not p.exists():
        return ScfStatus(
            iterations=0, converged=False, ef_ry=None, etot_ry=None,
            last_err=None, history=[],
        )
    return parse_scf_log(str(p))


def format_eta(seconds: float) -> str:
    """``3701`` -> ``"1h 1m"``; ``65`` -> ``"1m 5s"``; ``9`` -> ``"9s"``."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


@dataclass
class EtaModel:
    """t_point_s: seconds per (E, theta, phi) point per core, calibrated per host."""

    t_point_s: float = 4.0
    store_path: Optional[str] = None

    def estimate_seconds(self, params, nproc: int) -> float:
        nproc = max(1, int(nproc))
        n_points = int(getattr(params, "n_points"))
        return self.t_point_s * n_points / nproc

    def calibrate(self, params, nproc: int, wall_s: float, host: str = "local") -> float:
        nproc = max(1, int(nproc))
        n_points = int(getattr(params, "n_points"))
        if n_points <= 0:
            raise ValueError("params.n_points must be > 0 to calibrate ETA")
        self.t_point_s = float(wall_s) * nproc / n_points
        if self.store_path:
            self._save(host)
        return self.t_point_s

    def load(self, host: str = "local") -> float:
        """Load a previously-calibrated t_point_s for ``host`` from disk.

        No-op (returns current value) when store_path is None or the file /
        host entry does not exist yet.
        """
        if not self.store_path:
            return self.t_point_s
        p = Path(self.store_path).expanduser()
        if not p.exists():
            return self.t_point_s
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            return self.t_point_s
        entry = data.get(host)
        if isinstance(entry, dict) and "t_point_s" in entry:
            self.t_point_s = float(entry["t_point_s"])
        return self.t_point_s

    def _save(self, host: str) -> None:
        p = Path(self.store_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except (OSError, ValueError):
                data = {}
        data[host] = {"t_point_s": self.t_point_s}
        p.write_text(json.dumps(data, indent=2))
