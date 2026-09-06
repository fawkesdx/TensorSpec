"""End-to-end tests for the SSL training CLI."""

import json
from pathlib import Path

import numpy as np

from tensorspec.core.ml.ssl.cli import main
from tensorspec.core.ml.ssl.shards import ShardWriter, write_manifest
from tensorspec.core.ml.ssl.spec import (
    AugmentSpec,
    DinoSpec,
    ModelSpec,
    OptimSpec,
    RunConfig,
    to_jsonable,
)


def _mini_shards(tmp_path: Path, n: int = 16) -> None:
    writer = ShardWriter(str(tmp_path), target_bytes=10_000_000)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.random((32, 32), dtype=np.float32)
        writer.add(img, {"source_id": "t.h5", "index": {"y": i, "x": 0}})
    partial = writer.close()
    manifest = {
        **partial,
        "preprocess": {"sample": {"mode": "disp2d", "index_roles": ["y", "x"]}},
        "sources": [
            {
                "id": "t.h5",
                "size": 1,
                "sha256_head": "0" * 64,
                "sha256_tail": "0" * 64,
                "kind": "xy_fine_4d",
                "shape": [n, 1, 32, 32],
                "detector": {},
                "dead_pixel": {},
                "calibration": {
                    "deg_per_raw_px": 0.048,
                    "axis_source": "test",
                },
            }
        ],
    }
    write_manifest(str(tmp_path / "manifest.json"), manifest)


def _mini_run_config(max_steps: int = 3) -> RunConfig:
    return RunConfig(
        seed=0,
        max_steps=max_steps,
        log_every=1,
        ckpt_every=max_steps,
        num_workers=0,
        augment=AugmentSpec(
            arm="A1",
            n_global=2,
            n_local=2,
            global_size=32,
            local_size=24,
            global_crop_frac=1.0,
        ),
        model=ModelSpec(name="vit_ti", img_size=32, patch_size=8),
        dino=DinoSpec(out_dim=32, hidden_dim=32, bottleneck_dim=16),
        optim=OptimSpec(
            batch_size=4,
            epochs=1,
            warmup_epochs=0,
            use_amp=False,
            lr=1e-3,
        ),
    )


def test_cli_train_runs_and_writes_metrics(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _mini_shards(data)
    out = tmp_path / "run"
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(to_jsonable(_mini_run_config(max_steps=3))),
        encoding="utf-8",
    )

    result = main(
        [
            "train",
            "--config",
            str(config_path),
            "--data",
            str(data),
            "--out",
            str(out),
        ]
    )

    assert result == 0
    metrics_path = out / "metrics.jsonl"
    assert metrics_path.exists()
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2, 3]
    assert (out / "checkpoints" / "last.pt").exists()
