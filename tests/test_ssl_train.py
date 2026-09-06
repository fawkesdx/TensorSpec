import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorspec.core.ml.ssl.augment import MultiCropViews
from tensorspec.core.ml.ssl.shards import ShardWriter, write_manifest
from tensorspec.core.ml.ssl.spec import (
    AugmentSpec,
    DinoSpec,
    ModelSpec,
    OptimSpec,
    RunConfig,
)
from tensorspec.core.ml.ssl.train import (
    _average_loss_centers,
    _collate_views,
    _data_loader_workers,
    _load_resume_checkpoint,
    _remaining_batches,
    _resume_position,
    _schedule_values,
    _synchronize_bad_count,
    train,
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
    assert records[-1]["throughput"] > 0
    assert records[-1]["skipped_batches"] == 0
    assert records[-1]["bad_samples"] == 0
    assert records[-1]["nonfinite_losses"] == 0

    cfg.max_steps = 1
    train(cfg, str(data), str(out))
    fresh_records = (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in fresh_records] == [1]


def test_resume_position_skips_completed_batches() -> None:
    assert _resume_position(step=5, steps_per_epoch=4) == (1, 1)
    assert list(_remaining_batches(["batch-0", "batch-1", "batch-2"], 1)) == [
        "batch-1",
        "batch-2",
    ]


def test_single_process_resume_disables_data_loader_workers() -> None:
    assert _data_loader_workers(2, resume=True, distributed=False) == 0
    assert _data_loader_workers(2, resume=False, distributed=False) == 2
    assert _data_loader_workers(2, resume=True, distributed=True) == 2


def test_warmup_and_cosine_schedules_hit_endpoints() -> None:
    cfg = RunConfig(
        optim=OptimSpec(lr=0.004, warmup_epochs=1),
        dino=DinoSpec(
            teacher_temp_start=0.04,
            teacher_temp_end=0.07,
            teacher_temp_warmup_epochs=1,
        ),
    )

    first = _schedule_values(cfg, 0, steps_per_epoch=4, total_steps=8)
    warmup_last = _schedule_values(cfg, 3, steps_per_epoch=4, total_steps=8)
    final = _schedule_values(cfg, 7, steps_per_epoch=4, total_steps=8)

    assert first[0] == pytest.approx(0.001)
    assert warmup_last[0] == pytest.approx(0.004)
    assert final[0] == 0.0
    assert first[3] == pytest.approx(0.04)
    assert warmup_last[3] == pytest.approx(0.07)
    assert final[3] == pytest.approx(0.07)


def test_average_loss_centers_uses_all_rank_statistics() -> None:
    model = SimpleNamespace(
        dino_loss=SimpleNamespace(center=torch.tensor([[1.0]])),
        ibot_loss=SimpleNamespace(center=torch.tensor([[1.0]])),
    )

    def add_remote_center(center: torch.Tensor) -> None:
        center.add_(3.0)

    _average_loss_centers(model, world_size=2, all_reduce=add_remote_center)

    assert model.dino_loss.center.item() == 2.0
    assert model.ibot_loss.center.item() == 2.0


def test_collate_counts_invalid_source_samples() -> None:
    good = MultiCropViews(
        [torch.ones(1, 4, 4), torch.ones(1, 4, 4)], valid=True
    )
    bad = MultiCropViews(
        [torch.zeros(1, 4, 4), torch.zeros(1, 4, 4)], valid=False
    )

    views, invalid_count = _collate_views([good, bad])

    assert len(views) == 2
    assert views[0].shape == (2, 1, 4, 4)
    assert invalid_count == 1


def test_bad_count_is_summed_across_ranks() -> None:
    def add_remote_bad_count(value: torch.Tensor) -> None:
        value.add_(2)

    assert (
        _synchronize_bad_count(
            1,
            torch.device("cpu"),
            world_size=2,
            all_reduce=add_remote_bad_count,
        )
        == 3
    )


def test_resume_falls_back_to_previous_valid_step_checkpoint(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    valid_path = checkpoint_dir / "step_000001.pt"
    torch.save(
        {
            "step": 1,
            "epoch": 0,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": {},
            "seed": 0,
            "git_commit": "test",
        },
        valid_path,
    )
    corrupt_path = checkpoint_dir / "step_000002.pt"
    corrupt_path.write_bytes(b"not a torch checkpoint")

    with pytest.warns(RuntimeWarning, match="using"):
        checkpoint, loaded_path = _load_resume_checkpoint(
            corrupt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=torch.device("cpu"),
        )

    assert checkpoint["step"] == 1
    assert loaded_path == valid_path


def test_train_skips_nonfinite_loss_and_counts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _mini_shards(data, n=4)
    out = tmp_path / "run"
    cfg = RunConfig(
        seed=3,
        max_steps=2,
        log_every=1,
        ckpt_every=2,
        num_workers=0,
        augment=AugmentSpec(
            arm="A1",
            n_global=2,
            n_local=0,
            global_size=32,
            global_crop_frac=1.0,
        ),
        model=ModelSpec(name="vit_ti", img_size=32, patch_size=8),
        dino=DinoSpec(out_dim=16, hidden_dim=16, bottleneck_dim=8),
        optim=OptimSpec(
            batch_size=2,
            epochs=1,
            warmup_epochs=0,
            use_amp=False,
        ),
    )
    train_module = importlib.import_module("tensorspec.core.ml.ssl.train")
    original_forward = train_module._LossModule.forward
    calls = 0

    def one_bad_loss(self, views, teacher_temp):
        nonlocal calls
        loss, components = original_forward(self, views, teacher_temp)
        calls += 1
        if calls == 1:
            loss = loss * torch.tensor(float("nan"), device=loss.device)
        return loss, components

    monkeypatch.setattr(train_module._LossModule, "forward", one_bad_loss)

    summary = train(cfg, str(data), str(out))

    records = [
        json.loads(line)
        for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["steps"] == 2
    assert records[0]["skipped"]
    assert records[0]["nonfinite_losses"] == 1
    assert records[-1]["skipped_batches"] == 1
    assert np.isfinite(records[-1]["loss"])


def test_train_skips_zero_source_batch_and_counts_samples(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _mini_shards(data, n=4)
    shard_path = data / "shard_00000.npy"
    shard = np.load(shard_path, allow_pickle=False)
    np.save(shard_path, np.zeros_like(shard))
    out = tmp_path / "run"
    cfg = RunConfig(
        seed=5,
        max_steps=1,
        log_every=1,
        ckpt_every=1,
        num_workers=0,
        augment=AugmentSpec(
            arm="A1",
            n_global=2,
            n_local=0,
            global_size=32,
        ),
        model=ModelSpec(name="vit_ti", img_size=32, patch_size=8),
        dino=DinoSpec(out_dim=16, hidden_dim=16, bottleneck_dim=8),
        optim=OptimSpec(batch_size=4, epochs=1, use_amp=False),
    )

    summary = train(cfg, str(data), str(out))

    record = json.loads((out / "metrics.jsonl").read_text(encoding="utf-8"))
    assert summary["last_loss"] is None
    assert record["skipped"]
    assert record["skipped_batches"] == 1
    assert record["bad_samples"] == 4
