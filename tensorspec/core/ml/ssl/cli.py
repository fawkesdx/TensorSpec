"""Command-line entry point for SSL dataset preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from tensorspec.core.ml.ssl.preprocess import preprocess_file
from tensorspec.core.ml.ssl.spec import (
    NormSpec,
    PreprocessConfig,
    ResampleSpec,
    SampleSpec,
    TrimSpec,
    preprocess_config_from_dict,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensorspec.core.ml.ssl.cli")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    preprocess = subparsers.add_parser(
        "preprocess", help="convert one Maestro file into SSL shards"
    )
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--out", required=True)
    preprocess.add_argument(
        "--mode", choices=["disp2d", "fermi3d"], required=True
    )
    preprocess.add_argument(
        "--config", help="JSON PreprocessConfig; defaults used if omitted"
    )
    preprocess.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing shard dataset in --out",
    )
    preprocess.add_argument(
        "--append",
        action="store_true",
        help="append samples into an existing dataset in --out",
    )
    preprocess.add_argument(
        "--source-id",
        help="manifest source id (default: input basename)",
    )
    return parser


def _config(path: str | None, mode: str) -> PreprocessConfig:
    if path is None:
        config = PreprocessConfig(
            trim=TrimSpec(ranges={}, source_kind=""),
            norm=NormSpec(),
            resample=ResampleSpec(),
            sample=SampleSpec(mode=mode, index_roles=()),
        )
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = preprocess_config_from_dict(payload)
    return replace(config, sample=replace(config.sample, mode=mode))


def _progress_printer(label: str):
    started = time.monotonic()
    last_print = 0.0

    def progress(current: int, total: int) -> None:
        nonlocal last_print
        now = time.monotonic()
        done = current >= total
        if not done and now - last_print < 5.0 and current % 500 != 0:
            return
        last_print = now
        elapsed = now - started
        rate = current / elapsed if elapsed > 0 else 0.0
        print(
            f"{label}: {current}/{total} ({100.0 * current / max(total, 1):.1f}%) "
            f"{rate:.1f} samp/s",
            file=sys.stderr,
            flush=True,
        )

    return progress


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "preprocess":
        if args.overwrite and args.append:
            raise SystemExit("error: --overwrite and --append are mutually exclusive")
        label = Path(args.input).name
        manifest = preprocess_file(
            args.input,
            args.out,
            _config(args.config, args.mode),
            source_id=args.source_id,
            overwrite=args.overwrite,
            append=args.append,
            progress=_progress_printer(label),
        )
        print(
            f"done {label}: total_samples={manifest['total_samples']} "
            f"sources={len(manifest['sources'])}",
            file=sys.stderr,
            flush=True,
        )
        return 0
    raise AssertionError(f"unhandled command {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
