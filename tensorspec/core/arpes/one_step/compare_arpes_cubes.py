#!/usr/bin/env python3
"""Compare two ARPES intensity cubes (chinook vs GrizzlyME).

Expects ``.npz`` files from ``chinook_remote_runner`` with keys:
``cube``, and optionally ``energy``, ``theta``, ``phi``.

Examples
--------
    python compare_arpes_cubes.py chinook.npz grizzly.npz
    python compare_arpes_cubes.py a.npz b.npz --key cube --rtol 1e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_cube(path: Path, key: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"{path}: missing '{key}'. keys={list(data.keys())}")
    arr = np.asarray(data[key], dtype=np.float64)
    return arr


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) + 1e-30
    return float(np.linalg.norm(a - b) / denom)


def _summary(name: str, a: np.ndarray) -> None:
    print(
        f"  {name}: shape={a.shape}  min={a.min():.4e}  max={a.max():.4e}  "
        f"sum={a.sum():.4e}  mean={a.mean():.4e}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ref", type=Path, help="Reference cube (.npz), e.g. chinook")
    p.add_argument("cmp", type=Path, help="Comparison cube (.npz), e.g. GrizzlyME")
    p.add_argument("--key", default="cube", help="Array key inside npz (default: cube)")
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Pass if relative L2 <= this (default: 1e-3)",
    )
    p.add_argument(
        "--atol-max",
        type=float,
        default=None,
        help="Optional: also require max|Δ| <= this absolute value",
    )
    args = p.parse_args(argv)

    a = _load_cube(args.ref, args.key)
    b = _load_cube(args.cmp, args.key)

    print(f"ref: {args.ref}")
    print(f"cmp: {args.cmp}")
    _summary("ref", a)
    _summary("cmp", b)

    if a.shape != b.shape:
        print(f"FAIL: shape mismatch {a.shape} vs {b.shape}")
        return 2

    diff = a - b
    abs_diff = np.abs(diff)
    rel = _rel_l2(a, b)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    # Peak-normalized max error (common for maps)
    peak = float(max(a.max(), 1e-30))
    max_rel_peak = max_abs / peak

    print("---")
    print(f"rel L2 (||Δ|| / ||ref||):     {rel:.6e}")
    print(f"max |Δ|:                      {max_abs:.6e}")
    print(f"mean |Δ|:                     {mean_abs:.6e}")
    print(f"max |Δ| / max(ref):           {max_rel_peak:.6e}")

    ok = rel <= args.rtol
    if args.atol_max is not None:
        ok = ok and max_abs <= args.atol_max

    print("---")
    if ok:
        print(f"PASS (rel L2 <= {args.rtol:g}"
              + (f", max|Δ| <= {args.atol_max:g}" if args.atol_max is not None else "")
              + ")")
        return 0

    print(f"FAIL (rel L2 {rel:.3e} > {args.rtol:g}"
          + (f" or max|Δ| {max_abs:.3e} > {args.atol_max:g}" if args.atol_max is not None else "")
          + ")")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
