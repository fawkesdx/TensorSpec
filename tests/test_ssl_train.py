import json
from pathlib import Path

import numpy as np
import torch

from tensorspec.core.ml.ssl.shards import ShardWriter, write_manifest
from tensorspec.core.ml.ssl.spec import (
    AugmentSpec,
    DinoSpec,
    ModelSpec,
    OptimSpec,
    RunConfig,
)
from tensorspec.core.ml.ssl.train import train


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


def test_train_smoke_writes_checkpoint_metrics_and_resumes(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _mini_shards(data)
    out = tmp_path / "run"
    cfg = RunConfig(
        seed=0,
        max_steps=5,
        log_every=1,
        ckpt_every=5,
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

    summary = train(cfg, str(data), str(out))

    assert summary["steps"] == 5
    assert json.loads((out / "run_config.json").read_text())["max_steps"] == 5
    checkpoint_path = out / "checkpoints" / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint_path.exists()
    assert set(checkpoint) == {
        "step",
        "epoch",
        "model",
        "optimizer",
        "scaler",
        "config",
        "seed",
        "git_commit",
    }
    assert checkpoint["step"] == 5

    cfg.max_steps = 8
    resumed = train(cfg, str(data), str(out), resume=str(checkpoint_path))

    assert resumed["steps"] == 8
    records = [
        json.loads(line)
        for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in records] == list(range(1, 9))
    assert all(np.isfinite(record["loss"]) for record in records)
    assert records[-1]["lr"] == 0.0
    assert records[-1]["weight_decay"] == cfg.optim.weight_decay_end
    assert records[-1]["teacher_momentum"] == 1.0
