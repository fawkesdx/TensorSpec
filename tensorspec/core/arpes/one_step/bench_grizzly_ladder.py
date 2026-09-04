#!/usr/bin/env python3
"""Run scale-ladder rungs sequentially; append results to scale_bench/LADDER.jsonl.

Intended for remote ``chinook_gui_run``. Skips rungs whose result JSON exists
unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# (ntheta, nphi, ne, phi_min, phi_max, e_min, e_max, theta_chunk)
DEFAULT_RUNGS = [
    (80, 1, 80, 0.0, 0.0, -1.0, 0.1, 0),
    (80, 40, 40, -15.0, 15.0, -1.0, 0.1, 0),
    (80, 80, 40, -15.0, 15.0, -1.0, 0.1, 20),  # chunk if needed
    (100, 100, 40, -15.0, 15.0, -1.0, 0.1, 20),
    (200, 1, 200, 0.0, 0.0, -1.0, 0.1, 40),
    (200, 40, 100, -5.0, 5.0, -1.0, 0.1, 20),
]


def tag(nt, np_, ne):
    return f"{nt}x{np_}x{ne}"


def result_path(out_dir: Path, nt, np_, ne, engine, device, layout) -> Path:
    return out_dir / f"result_{tag(nt, np_, ne)}_{engine}_{device}_{layout}.json"


def parse_log(text: str) -> dict:
    out = {
        "completed": "completed successfully" in text,
        "oom": bool(re.search(r"out of memory|CUDA OOM", text, re.I)),
        "full_cube_wall_s": None,
        "layout_used": None,
        "theta_chunk_used": None,
    }
    m = re.search(r"full-cube wall:\s*([0-9.]+)s", text)
    if m:
        out["full_cube_wall_s"] = float(m.group(1))
    m = re.search(
        r"completed successfully \(engine=([^,]+), layout=([^,\)]+)(?:, theta_chunk=([0-9]+))?\)",
        text,
    )
    if m:
        out["layout_used"] = m.group(2).strip()
        if m.group(3):
            out["theta_chunk_used"] = int(m.group(3))
    return out


def run_one(py: str, out_dir: Path, rung, force: bool) -> dict:
    nt, nphi, ne, phi0, phi1, e0, e1, tchunk = rung
    engine, device, layout = "grizzly", "cuda", "full"
    rp = result_path(out_dir, nt, nphi, ne, engine, device, layout)
    if rp.is_file() and not force:
        print(f"SKIP existing {rp.name}", flush=True)
        return json.loads(rp.read_text())

    cmd = [
        py,
        "-u",
        "bench_grizzly_scale.py",
        "--ntheta",
        str(nt),
        "--nphi",
        str(nphi),
        "--ne",
        str(ne),
        "--phi_min",
        str(phi0),
        "--phi_max",
        str(phi1),
        "--e_min",
        str(e0),
        "--e_max",
        str(e1),
        "--device",
        device,
        "--layout",
        layout,
        "--theta_chunk",
        str(tchunk),
        "--cuda_visible",
        "0",
        "--out_dir",
        str(out_dir),
    ]
    print(f"RUN {tag(nt, nphi, ne)} theta_chunk={tchunk}", flush=True)
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    # Foreground wait via bench (no --background)
    proc = subprocess.run(cmd, env=env)
    wall = time.perf_counter() - t0
    log = out_dir / f"log_{tag(nt, nphi, ne)}_{engine}_{device}_{layout}.txt"
    text = log.read_text(errors="replace") if log.is_file() else ""
    parsed = parse_log(text)
    cube = out_dir / f"cube_{tag(nt, nphi, ne)}_{engine}_{device}_{layout}.npz"
    result = {
        "tag": tag(nt, nphi, ne),
        "ntheta": nt,
        "nphi": nphi,
        "ne": ne,
        "nk": nt * nphi,
        "theta_chunk_requested": tchunk,
        "exit_code": proc.returncode,
        "wall_s": wall,
        "cube_exists": cube.is_file(),
        "cube_bytes": cube.stat().st_size if cube.is_file() else 0,
        **parsed,
    }
    rp.write_text(json.dumps(result, indent=2))
    with open(out_dir / "LADDER.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="scale_bench")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--from_tag",
        default="",
        help="Skip rungs until this tag (e.g. 80x40x40)",
    )
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = not args.from_tag
    for rung in DEFAULT_RUNGS:
        t = tag(rung[0], rung[1], rung[2])
        if not started:
            if t == args.from_tag:
                started = True
            else:
                print(f"SKIP until {args.from_tag}: {t}", flush=True)
                continue
        result = run_one(args.python, out_dir, rung, args.force)
        if result.get("exit_code", 1) != 0 and not result.get("completed"):
            print(f"HARD WALL at {t}; stopping ladder.", flush=True)
            return 1
        if result.get("oom") and result.get("layout_used") == "slices":
            print(f"WALL: fell back to slices at {t}", flush=True)
            # continue — still useful data
    print("LADDER complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
