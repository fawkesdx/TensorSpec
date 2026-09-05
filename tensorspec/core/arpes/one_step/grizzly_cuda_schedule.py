"""
CUDA device discovery and dynamic θ-chunk planning for Grizzly full-cube ARPES.

Estimates per-block VRAM from TB size (hoppings, basis) and grid (φ, E), then
picks the largest θ-chunk that fits free GPU memory. Multi-GPU runs assign
θ-blocks to workers via a shared queue (one OS process per GPU, spawn-safe).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CudaDeviceInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int


def probe_cuda_devices() -> List[CudaDeviceInfo]:
    """Return visible CUDA devices with current free/total VRAM."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out: List[CudaDeviceInfo] = []
    for i in range(torch.cuda.device_count()):
        try:
            free_b, total_b = torch.cuda.mem_get_info(i)
            props = torch.cuda.get_device_properties(i)
            out.append(
                CudaDeviceInfo(
                    index=i,
                    name=props.name,
                    total_bytes=int(total_b),
                    free_bytes=int(free_b),
                )
            )
        except Exception:
            continue
    return out


def resolve_gpu_ids(ngpus: int, devices: Optional[Sequence[CudaDeviceInfo]] = None) -> List[int]:
    """
    *ngpus*:
      0 / negative → all visible devices
      N>0 → first N device indices
    """
    ids, _, _ = select_gpu_ids(ngpus, devices)
    return ids


def select_gpu_ids(
    ngpus: int,
    devices: Optional[Sequence[CudaDeviceInfo]] = None,
    *,
    min_free_bytes: int = 10 * 1024**3,
) -> Tuple[List[int], int, List[str]]:
    """
    Pick GPU indices for a job with safety clamps.

    Returns (gpu_ids, effective_ngpus, warning_messages).
    Prefers devices with the most free VRAM; skips tight GPUs when alternatives exist.
    """
    devs = list(devices) if devices is not None else probe_cuda_devices()
    warnings: List[str] = []
    if not devs:
        return [], 0, warnings

    n_visible = len(devs)
    requested = n_visible if ngpus <= 0 or ngpus >= n_visible else max(1, int(ngpus))

    ranked = sorted(devs, key=lambda d: d.free_bytes, reverse=True)
    roomy = [d for d in ranked if d.free_bytes >= min_free_bytes]

    if requested == 1:
        pick = roomy[0] if roomy else ranked[0]
        if pick.free_bytes < min_free_bytes:
            warnings.append(
                f"GPU {pick.index} has only {pick.free_bytes / 1024**3:.1f} GiB free "
                f"(<{min_free_bytes / 1024**3:.0f} GiB guideline); may OOM."
            )
        return [pick.index], 1, warnings

    if len(roomy) >= requested:
        pool = roomy
    elif roomy:
        warnings.append(
            f"Only {len(roomy)} GPU(s) have >={min_free_bytes / 1024**3:.0f} GiB free; "
            f"using {len(roomy)} instead of {requested}."
        )
        pool = roomy
    else:
        warnings.append(
            f"No GPU has >={min_free_bytes / 1024**3:.0f} GiB free; best-effort on top {requested}."
        )
        pool = ranked

    effective = min(requested, len(pool))
    if effective < requested:
        warnings.append(f"Clamped GPU request {requested} -> {effective} (visible/usable).")

    ids = [pool[i].index for i in range(effective)]
    return ids, effective, warnings


def tb_hopping_stats(tb_model: Any) -> Tuple[int, int]:
    n_hops = sum(len(me.H) for me in tb_model.mat_els)
    n_basis = int(len(tb_model.basis))
    return max(int(n_hops), 1), max(n_basis, 1)


def inner_diag_k_batch_bytes(n_hops: int, n_basis: int, phase_budget_bytes: int) -> int:
    """Peak VRAM for one inner Fourier+diag batch (hop-limited)."""
    hop_chunk = max(1, phase_budget_bytes // (n_hops * 16))
    phase = hop_chunk * n_hops * 16
    hb = hop_chunk * n_basis * n_basis * 16 * 2  # H + eigenvectors
    return phase + hb


def estimate_theta_block_bytes(
    ntheta_block: int,
    nphi: int,
    ne: int,
    n_hops: int,
    n_basis: int,
    *,
    phase_budget_bytes: int,
) -> int:
    """
    Rough peak VRAM for one Grizzly θ-block on GPU.

    Empirical: ~5.25 MiB per k-point (V100, ~6.7M-hop TB, ne~200).
  """
    nk = max(1, ntheta_block * nphi)
    per_k_bytes = 5_250_000
    return int(nk * per_k_bytes + 400 * 1024**2)


def phase_budget_from_free(free_bytes: int, safety: float = 0.35) -> int:
    """Bytes reserved for inner diag Fourier phases per batch."""
    return max(256 * 1024**2, int(free_bytes * safety))


def plan_theta_chunk(
    tb_model: Any,
    ntheta: int,
    nphi: int,
    ne: int,
    device: Optional[CudaDeviceInfo],
    *,
    safety: float = 0.65,
    min_chunk: int = 1,
) -> int:
    """
    Largest θ-chunk (≤ *ntheta*) whose estimated peak VRAM fits *device* free memory.

    Returns *min_chunk* if no CUDA info (CPU path).
    """
    n_hops, n_basis = tb_hopping_stats(tb_model)
    if device is None:
        return max(min_chunk, min(ntheta, 20))

    phase_budget = phase_budget_from_free(device.free_bytes)
    budget = int(device.free_bytes * safety)

    lo, hi = min_chunk, ntheta
    best = min_chunk
    while lo <= hi:
        mid = (lo + hi) // 2
        need = estimate_theta_block_bytes(
            mid, nphi, ne, n_hops, n_basis, phase_budget_bytes=phase_budget
        )
        if need <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return max(min_chunk, min(ntheta, best))


def shrink_chunk_schedule(
    first_chunk: int,
    ntheta: int,
    *,
    include_one_shot: bool = False,
) -> List[int]:
    """OOM retry order: try *first_chunk*, then successive halves down to 1."""
    sched: List[int] = []
    if include_one_shot and first_chunk <= 0:
        sched.append(0)
    c = max(1, first_chunk) if first_chunk > 0 else max(1, ntheta // 2)
    while c >= 1:
        if c not in sched and c <= ntheta:
            sched.append(c)
        if c == 1:
            break
        c = max(1, c // 2)
    return sched


def format_device_summary(devices: Sequence[CudaDeviceInfo]) -> str:
    if not devices:
        return "no CUDA devices"
    parts = []
    for d in devices:
        parts.append(
            f"gpu{d.index}={d.name} "
            f"free={d.free_bytes / 1024**3:.1f}GiB/"
            f"total={d.total_bytes / 1024**3:.1f}GiB"
        )
    return "; ".join(parts)
