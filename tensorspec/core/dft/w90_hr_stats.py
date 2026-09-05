"""Quick wannier90_hr.dat size / complexity checks before band diag."""

from __future__ import annotations

import os
from typing import Dict, Optional


def quick_w90_hr_stats(w90_filepath: str) -> Dict[str, float]:
    """Estimate TB cost from hr.dat header (matches chinook_tb parser layout)."""
    path = os.path.abspath(w90_filepath)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    num_wann = 0
    nrpts = 0
    hop_rows = 0
    try:
        with open(path, "r") as f:
            f.readline()
            num_wann = int(f.readline().strip())
            nrpts = int(f.readline().strip())
            deg_lines = int(__import__("numpy").ceil(nrpts / 15.0))
            data_start = 3 + deg_lines
        with open(path, "r") as f:
            hop_rows = sum(1 for _ in f) - data_start
    except Exception:
        hop_rows = int(size_mb * 1e6 / 80)  # ~80 bytes/row fallback

    est_hops = max(hop_rows, nrpts * num_wann * num_wann if num_wann and nrpts else 0)
    return {
        "size_mb": size_mb,
        "num_wann": float(num_wann),
        "nrpts": float(nrpts),
        "hop_rows": float(max(hop_rows, 0)),
        "est_hops": float(est_hops),
    }


def format_w90_cost_warning(stats: Dict[str, float], nk: int) -> Optional[str]:
    """Return user-facing warning text, or None if job looks small."""
    size_mb = stats.get("size_mb", 0)
    est_hops = stats.get("est_hops", 0)
    num_wann = int(stats.get("num_wann", 0))
    if size_mb < 30 and est_hops < 500_000:
        return None
    lines = [
        f"Large Wannier90 TB detected:",
        f"  hr.dat ≈ {size_mb:.0f} MB, ~{est_hops/1e6:.1f}M hop rows, {num_wann} Wannier bands",
        f"  k-points this run: {nk}",
    ]
    if est_hops > 5_000_000:
        lines.append(
            "This can take hours on CPU/GPU — Wannier90 export is likely untruncated "
            "(too many R-vectors)."
        )
        lines.append(
            "Faster options: re-run Wannier with tighter hr_plot / disentanglement, "
            "fewer k-path points, Hybrid + cached parse, or band diag without fat-band projection."
        )
    elif est_hops > 500_000:
        lines.append("Expect minutes, not seconds. Hybrid mode recommended.")
    return "\n".join(lines)
