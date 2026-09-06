"""Stage-1 DINO training loop with metrics and checkpoint resume."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Callable, Iterable, Iterator, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tensorspec.core.ml.ssl.augment import MultiCropDataset
from tensorspec.core.ml.ssl.dino import DinoModel, dino_total_loss, update_teacher
from tensorspec.core.ml.ssl.models.vit2d import build_vit2d
from tensorspec.core.ml.ssl.shards import ShardDataset
from tensorspec.core.ml.ssl.spec import RunConfig, to_jsonable

_Batch = TypeVar("_Batch")


class _LossModule(nn.Module):
    """Give DDP a conventional forward while retaining the DinoModel API."""

    def __init__(self, model: DinoModel, config: RunConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def forward(
        self, views: list[torch.Tensor], teacher_temp: float
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return dino_total_loss(
            self.model,
            views,
            self.config.dino,
            teacher_temp=teacher_temp,
            mask_ratio=self.config.dino.ibot_mask_ratio,
        )


def _collate_views(samples: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    return [
        torch.stack([sample[view_index] for sample in samples])
        for view_index in range(len(samples[0]))
    ]


def _resume_position(step: int, steps_per_epoch: int) -> tuple[int, int]:
    return divmod(step, steps_per_epoch)


def _data_loader_workers(
    configured_workers: int, *, resume: bool, distributed: bool
) -> int:
    # Keep single-process resume independent of prefetched worker queues.
    return 0 if resume and not distributed else configured_workers


def _remaining_batches(
    batches: Iterable[_Batch], completed_batches: int
) -> Iterator[_Batch]:
    for batch_index, batch in enumerate(batches):
        if batch_index >= completed_batches:
            yield batch


@torch.no_grad()
def _average_loss_centers(
    model: DinoModel,
    world_size: int,
    all_reduce: Callable[[torch.Tensor], Any] | None = None,
) -> None:
    if world_size <= 1:
        return
    reduce = dist.all_reduce if all_reduce is None else all_reduce
    for loss_module in (model.dino_loss, model.ibot_loss):
        reduce(loss_module.center)
        loss_module.center.div_(world_size)


def _cosine(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def _schedule_values(
    config: RunConfig,
    step_index: int,
    steps_per_epoch: int,
    total_steps: int,
) -> tuple[float, float, float, float]:
    warmup_steps = int(config.optim.warmup_epochs * steps_per_epoch)
    if warmup_steps and step_index < warmup_steps:
        lr = config.optim.lr * (step_index + 1) / warmup_steps
    else:
        cosine_steps = max(1, total_steps - warmup_steps - 1)
        progress = (step_index - warmup_steps) / cosine_steps
        lr = _cosine(config.optim.lr, 0.0, progress)

    progress = step_index / max(1, total_steps - 1)
    weight_decay = _cosine(
        config.optim.weight_decay_start,
        config.optim.weight_decay_end,
        progress,
    )
    momentum = _cosine(config.dino.momentum_teacher, 1.0, progress)

    temperature_warmup = int(
        config.dino.teacher_temp_warmup_epochs * steps_per_epoch
    )
    if temperature_warmup and step_index < temperature_warmup:
        temp_progress = step_index / max(1, temperature_warmup - 1)
        teacher_temp = (
            config.dino.teacher_temp_start
            + (config.dino.teacher_temp_end - config.dino.teacher_temp_start)
            * temp_progress
        )
    else:
        teacher_temp = config.dino.teacher_temp_end
    return lr, weight_decay, momentum, teacher_temp


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    epoch: int,
    model: DinoModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: RunConfig,
) -> dict[str, Any]:
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": to_jsonable(config),
        "seed": config.seed,
        "git_commit": _git_commit(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return checkpoint


def _device_and_ddp() -> tuple[torch.device, bool, int, int, bool]:
    distributed = "WORLD_SIZE" in os.environ
    initialized_here = False
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True

    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device, distributed, rank, world_size, initialized_here


def train(
    config: RunConfig,
    data_dir: str,
    out_dir: str,
    *,
    resume: str | None = None,
) -> dict[str, Any]:
    """Train DINO from shards and return final step/epoch summary."""
    device, distributed, rank, world_size, initialized_here = _device_and_ddp()
    is_main = rank == 0
    output = Path(out_dir)
    checkpoints = output / "checkpoints"
    metrics_path = output / "metrics.jsonl"

    try:
        if is_main:
            checkpoints.mkdir(parents=True, exist_ok=True)
            (output / "run_config.json").write_text(
                json.dumps(to_jsonable(config), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if resume is None:
                metrics_path.write_text("", encoding="utf-8")
        if distributed:
            dist.barrier()

        seed = config.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        dataset = MultiCropDataset(
            ShardDataset(data_dir), config.augment, seed=config.seed
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=config.seed,
            )
            if distributed
            else None
        )
        loader_generator = torch.Generator()
        loader = DataLoader(
            dataset,
            batch_size=config.optim.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=_data_loader_workers(
                config.num_workers,
                resume=resume is not None,
                distributed=distributed,
            ),
            drop_last=False,
            collate_fn=_collate_views,
            generator=loader_generator,
        )
        if not len(loader):
            raise ValueError("training dataset is empty")
        steps_per_epoch = len(loader)
        total_steps = (
            config.max_steps
            if config.max_steps is not None
            else config.optim.epochs * steps_per_epoch
        )

        student = build_vit2d(config.model)
        teacher = build_vit2d(config.model)
        teacher.load_state_dict(student.state_dict())
        model = DinoModel(student, teacher, config.dino).to(device)
        optimizer = torch.optim.AdamW(
            model.student_parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay_start,
        )
        amp_enabled = bool(config.optim.use_amp and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        step = 0
        if resume is not None:
            checkpoint = torch.load(resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            step = int(checkpoint["step"])
        epoch, completed_batches = _resume_position(step, steps_per_epoch)

        loss_module: nn.Module = _LossModule(model, config)
        if distributed:
            ddp_kwargs: dict[str, Any] = {}
            if device.type == "cuda":
                ddp_kwargs = {"device_ids": [device.index], "output_device": device.index}
            loss_module = DistributedDataParallel(loss_module, **ddp_kwargs)

        last_loss: float | None = None
        while step < total_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            else:
                loader_generator.manual_seed(config.seed + epoch)
            for views in _remaining_batches(loader, completed_batches):
                if step >= total_steps:
                    break
                views = [
                    F.interpolate(
                        view.to(device),
                        size=(config.model.img_size, config.model.img_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if view.shape[-2:] != (
                        config.model.img_size,
                        config.model.img_size,
                    )
                    else view.to(device)
                    for view in views
                ]
                lr, weight_decay, momentum, teacher_temp = _schedule_values(
                    config, step, steps_per_epoch, total_steps
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay

                optimizer.zero_grad(set_to_none=True)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast:
                    loss, components = loss_module(views, teacher_temp)
                _average_loss_centers(model, world_size)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.optim.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.student_parameters()), config.optim.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                update_teacher(model, momentum=momentum)

                step += 1
                last_loss = float(loss.detach())
                if is_main and (step % config.log_every == 0 or step == total_steps):
                    record = {
                        "step": step,
                        "epoch": epoch,
                        "loss": last_loss,
                        **{
                            name: float(value)
                            for name, value in components.items()
                        },
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "teacher_momentum": momentum,
                        "teacher_temp": teacher_temp,
                    }
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                if is_main and step % config.ckpt_every == 0:
                    checkpoint_epoch, _ = _resume_position(step, steps_per_epoch)
                    checkpoint = _save_checkpoint(
                        checkpoints / f"step_{step:06d}.pt",
                        step=step,
                        epoch=checkpoint_epoch,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                    )
                    last_path = checkpoints / "last.pt"
                    temporary = last_path.with_name(".last.pt.tmp")
                    torch.save(checkpoint, temporary)
                    temporary.replace(last_path)
            epoch += 1
            completed_batches = 0

        if is_main:
            checkpoint_epoch, _ = _resume_position(step, steps_per_epoch)
            checkpoint = _save_checkpoint(
                checkpoints / f"step_{step:06d}.pt",
                step=step,
                epoch=checkpoint_epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
            )
            temporary = checkpoints / ".last.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoints / "last.pt")
        if distributed:
            dist.barrier()
        return {
            "steps": step,
            "epoch": _resume_position(step, steps_per_epoch)[0],
            "last_loss": last_loss,
            "rank": rank,
            "world_size": world_size,
        }
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
