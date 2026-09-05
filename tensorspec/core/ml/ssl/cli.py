"""Command-line entry point for SSL dataset preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

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


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "preprocess":
        preprocess_file(args.input, args.out, _config(args.config, args.mode))
        return 0
    raise AssertionError(f"unhandled command {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
