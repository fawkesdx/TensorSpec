#!/usr/bin/env python3
"""Module 1 — mid-grid Grizzly scale probe (remote run dir).

Does NOT touch GUI default ``chinook_arpes_cube.npz``. Writes cube + JSON
under ``scale_bench/`` so interactive TensorSpec Fetch stays safe.

Run on the remote host inside the GUI run directory (tb_data.npz + physics present)::

    python -u bench_grizzly_scale.py --ntheta 40 --nphi 10 --ne 40 \\
        --device cuda --layout full

Or only print the nohup command (no launch)::

    python bench_grizzly_scale.py --ntheta 40 --nphi 40 --ne 40 --dry_run
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


def _grid_tag(ntheta: int, nphi: int, ne: int) -> str:
    return f"{ntheta}x{nphi}x{ne}"


def _parse_log(log_text: str) -> dict:
    out = {
        "completed": "completed successfully" in log_text,
        "oom": bool(
            re.search(r"out of memory|CUDA OOM|falling back to --layout slices", log_text, re.I)
        ),
        "full_cube_wall_s": None,
        "engine": None,
        "layout_used": None,
        "device": None,
    }
    m = re.search(r"full-cube wall:\s*([0-9.]+)s", log_text)
    if m:
        out["full_cube_wall_s"] = float(m.group(1))
    m = re.search(
        r"completed successfully \(engine=([^,]+), layout=([^)]+)\)", log_text
    )
    if m:
        out["engine"] = m.group(1).strip()
        out["layout_used"] = m.group(2).strip()
    m = re.search(r"Starting ARPES map on cluster \(([^)]+)\)", log_text)
    if m:
        blob = m.group(1)
        dm = re.search(r"device=([^,\s]+)", blob)
        if dm:
            out["device"] = dm.group(1)
        if out["engine"] is None:
            em = re.search(r"engine=([^,\s]+)", blob)
            if em:
                out["engine"] = em.group(1)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ntheta", type=int, required=True)
    p.add_argument("--nphi", type=int, required=True)
    p.add_argument("--ne", type=int, required=True)
    p.add_argument("--theta_min", type=float, default=-15.0)
    p.add_argument("--theta_max", type=float, default=15.0)
    p.add_argument("--phi_min", type=float, default=-15.0)
    p.add_argument("--phi_max", type=float, default=15.0)
    p.add_argument("--e_min", type=float, default=-1.0)
    p.add_argument("--e_max", type=float, default=0.1)
    p.add_argument("--hv", type=float, default=84.0)
    p.add_argument("--workf", type=float, default=4.5)
    p.add_argument("--v0", type=float, default=12.0)
    p.add_argument("--temp", type=float, default=10.0)
    p.add_argument("--engine", default="grizzly", choices=("grizzly", "chinook"))
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    p.add_argument("--layout", default="full", choices=("full", "slices", "auto"))
    p.add_argument("--theta_chunk", type=int, default=0)
    p.add_argument("--e_fermi", type=float, default=0.0)
    p.add_argument("--tb_file", default="tb_data.npz")
    p.add_argument("--physics_file", default="arpes_physics.json")
    p.add_argument("--out_dir", default="scale_bench")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--cuda_visible", default="0", help="CUDA_VISIBLE_DEVICES for this probe")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--background",
        action="store_true",
        help="nohup launch and return immediately (default: wait)",
    )
    args = p.parse_args()

    tag = _grid_tag(args.ntheta, args.nphi, args.ne)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cube_path = out_dir / f"cube_{tag}_{args.engine}_{args.device}_{args.layout}.npz"
    log_path = out_dir / f"log_{tag}_{args.engine}_{args.device}_{args.layout}.txt"
    json_path = out_dir / f"result_{tag}_{args.engine}_{args.device}_{args.layout}.json"

    cmd = [
        args.python,
        "-u",
        "chinook_remote_runner.py",
        "--tb_file",
        args.tb_file,
        "--physics_file",
        args.physics_file,
        "--theta_min",
        str(args.theta_min),
        "--theta_max",
        str(args.theta_max),
        "--ntheta",
        str(args.ntheta),
        "--phi_min",
        str(args.phi_min),
        "--phi_max",
        str(args.phi_max),
        "--nphi",
        str(args.nphi),
        "--e_min",
        str(args.e_min),
        "--e_max",
        str(args.e_max),
        "--ne",
        str(args.ne),
        "--hv",
        str(args.hv),
        "--workf",
        str(args.workf),
        "--v0",
        str(args.v0),
        "--temp",
        str(args.temp),
        "--polar",
        "P",
        "--cores",
        "1",
        "--engine",
        args.engine,
        "--device",
        args.device,
        "--layout",
        args.layout,
        "--theta_chunk",
        str(args.theta_chunk),
        "--e_fermi",
        str(args.e_fermi),
        "--out_file",
        str(cube_path),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    meta = {
        "tag": tag,
        "cmd": cmd,
        "cuda_visible": args.cuda_visible,
        "cube_path": str(cube_path),
        "log_path": str(log_path),
    }
    print(json.dumps({"launch": meta}, indent=2), flush=True)

    if args.dry_run:
        return 0

    if args.background:
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"PID={proc.pid} log={log_path}", flush=True)
        (out_dir / f"pid_{tag}.txt").write_text(str(proc.pid))
        return 0

    t0 = time.perf_counter()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    log_text = log_path.read_text(errors="replace")
    parsed = _parse_log(log_text)
    result = {
        **meta,
        "exit_code": proc.returncode,
        "wall_s": wall,
        **parsed,
        "cube_exists": cube_path.is_file(),
        "cube_bytes": cube_path.stat().st_size if cube_path.is_file() else 0,
    }
    json_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
