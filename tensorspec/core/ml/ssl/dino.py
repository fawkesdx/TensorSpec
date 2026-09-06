"""DINO, iBOT, and KoLeo objectives for Stage-1 SSL training."""

from __future__ import annotations

from itertools import chain
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorspec.core.ml.ssl.spec import DinoSpec


class DINOHead(nn.Module):
    """Three-layer projection head with normalized bottleneck and classifier."""

    def __init__(self, in_dim: int, spec: DinoSpec) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, spec.bottleneck_dim),
        )
        self.last_weight = nn.Parameter(
            torch.empty(spec.out_dim, spec.bottleneck_dim)
        )
        nn.init.trunc_normal_(self.last_weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1)
        weight = F.normalize(self.last_weight, dim=-1)
        return F.linear(x, weight)


def _loss_options(
    spec_or_out_dim: DinoSpec | int,
    student_temp: float | None,
    teacher_temp: float | None,
    center_momentum: float | None,
) -> tuple[int, float, float, float]:
    if isinstance(spec_or_out_dim, DinoSpec):
        spec = spec_or_out_dim
        return (
            spec.out_dim,
            spec.student_temp if student_temp is None else student_temp,
            spec.teacher_temp_start if teacher_temp is None else teacher_temp,
            spec.center_momentum if center_momentum is None else center_momentum,
        )
    return (
        int(spec_or_out_dim),
        0.1 if student_temp is None else student_temp,
        0.04 if teacher_temp is None else teacher_temp,
        0.9 if center_momentum is None else center_momentum,
    )


class DinoLoss(nn.Module):
    """Cross-view distillation loss with an fp32 running teacher center."""

    def __init__(
        self,
        spec_or_out_dim: DinoSpec | int,
        *,
        student_temp: float | None = None,
        teacher_temp: float | None = None,
        center_momentum: float | None = None,
    ) -> None:
        super().__init__()
        out_dim, student_temp, teacher_temp, center_momentum = _loss_options(
            spec_or_out_dim, student_temp, teacher_temp, center_momentum
        )
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim, dtype=torch.float32))

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        *,
        teacher_temp: float | None = None,
    ) -> torch.Tensor:
        student_fp32 = student_logits.float()
        teacher_fp32 = teacher_logits.detach().float()
        temperature = self.teacher_temp if teacher_temp is None else teacher_temp
        student_log_probs = F.log_softmax(
            student_fp32 / self.student_temp, dim=-1
        )
        teacher_probs = F.softmax(
            (teacher_fp32 - self.center) / temperature, dim=-1
        )
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        self._update_center(teacher_fp32)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_center = teacher_logits.mean(dim=tuple(range(teacher_logits.ndim - 1)))
        batch_center = batch_center.reshape_as(self.center).float()
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum
        )


class iBOTPatchLoss(DinoLoss):
    """Patch-token distillation loss over selected masked positions."""

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        teacher_temp: float | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
        if student_logits.numel() == 0:
            return student_logits.float().sum()
        return super().forward(
            student_logits, teacher_logits, teacher_temp=teacher_temp
        )


class KoLeoLoss(nn.Module):
    """Nearest-neighbor entropy regularizer from normalized CLS features."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] < 2:
            return features.float().sum() * 0.0
        normalized = F.normalize(features.float(), dim=-1, eps=self.eps)
        distances = torch.cdist(normalized, normalized, p=2)
        diagonal = torch.eye(
            distances.shape[0], dtype=torch.bool, device=distances.device
        )
        distances = distances.masked_fill(diagonal, float("inf"))
        nearest = distances.min(dim=1).values
        return -torch.log(nearest.clamp_min(self.eps)).mean()


class DinoModel(nn.Module):
    """Student/teacher backbones and their global and patch projection heads."""

    def __init__(
        self,
        student_backbone: nn.Module,
        teacher_backbone: nn.Module,
        spec: DinoSpec,
    ) -> None:
        super().__init__()
        in_dim = _backbone_dim(student_backbone)
        teacher_dim = _backbone_dim(teacher_backbone)
        if in_dim != teacher_dim:
            raise ValueError("student and teacher backbone dimensions must match")

        self.student = student_backbone
        self.teacher = teacher_backbone
        self.student_head = DINOHead(in_dim, spec)
        self.teacher_head = DINOHead(in_dim, spec)
        self.student_ibot_head = DINOHead(in_dim, spec)
        self.teacher_ibot_head = DINOHead(in_dim, spec)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())

        self.dino_loss = DinoLoss(spec)
        self.ibot_loss = iBOTPatchLoss(spec)
        self.koleo_loss = KoLeoLoss()
        self.spec = spec
        for parameter in self.teacher_parameters():
            parameter.requires_grad_(False)

    def forward_global(
        self, x: torch.Tensor, *, teacher: bool = False
    ) -> torch.Tensor:
        backbone = self.teacher if teacher else self.student
        head = self.teacher_head if teacher else self.student_head
        cls, _ = backbone.forward_features(x)
        return head(cls)

    def forward_ibot(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        teacher: bool = False,
    ) -> torch.Tensor:
        backbone = self.teacher if teacher else self.student
        head = self.teacher_ibot_head if teacher else self.student_ibot_head
        _, patches = backbone.forward_features(x, mask=mask)
        return head(patches)

    def student_parameters(self) -> Iterable[nn.Parameter]:
        return chain(
            self.student.parameters(),
            self.student_head.parameters(),
            self.student_ibot_head.parameters(),
        )

    def teacher_parameters(self) -> Iterable[nn.Parameter]:
        return chain(
            self.teacher.parameters(),
            self.teacher_head.parameters(),
            self.teacher_ibot_head.parameters(),
        )


def _backbone_dim(backbone: nn.Module) -> int:
    for name in ("embed_dim", "num_features"):
        value = getattr(backbone, name, None)
        if value is not None:
            return int(value)
    raise ValueError("backbone must expose embed_dim or num_features")


@torch.no_grad()
def update_teacher(
    student: DinoModel | nn.Module,
    teacher: nn.Module | None = None,
    m: float = 0.996,
    *,
    momentum: float | None = None,
) -> None:
    """EMA-update a DinoModel teacher or an explicit teacher module."""
    if momentum is not None:
        m = momentum
    if not 0.0 <= m <= 1.0:
        raise ValueError("teacher momentum must be in [0, 1]")

    if isinstance(student, DinoModel):
        if teacher is not None:
            raise TypeError("teacher must be omitted when updating a DinoModel")
        pairs = zip(student.student_parameters(), student.teacher_parameters())
    else:
        if teacher is None:
            raise TypeError("teacher module is required")
        pairs = zip(student.parameters(), teacher.parameters())

    for student_parameter, teacher_parameter in pairs:
        teacher_parameter.mul_(m).add_(student_parameter.detach(), alpha=1.0 - m)


def gram_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor | None = None,
    *,
    gram_enabled: bool = False,
) -> torch.Tensor:
    """Plan-B Gram placeholder; disabled by default."""
    if gram_enabled:
        raise NotImplementedError("Gram loss is not implemented in Plan B")
    return student_features.float().new_zeros(())


def _resize_for_backbone(view: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
    size = getattr(backbone, "img_size", None)
    if size is None or view.shape[-2:] == (size, size):
        return view
    return F.interpolate(view, size=(size, size), mode="bilinear", align_corners=False)


def _patch_mask(
    batch_size: int, patch_count: int, ratio: float, device: torch.device
) -> torch.Tensor:
    selected = min(patch_count, max(0, round(patch_count * ratio)))
    mask = torch.zeros(batch_size, patch_count, dtype=torch.bool, device=device)
    if selected:
        random_order = torch.rand(batch_size, patch_count, device=device).argsort(dim=1)
        mask.scatter_(1, random_order[:, :selected], True)
    return mask


def dino_total_loss(
    model: DinoModel,
    views: list[torch.Tensor],
    spec: DinoSpec,
    *,
    teacher_temp: float = 0.04,
    mask_ratio: float = 0.3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute global DINO, patch iBOT, KoLeo, and optional Gram losses."""
    if len(views) < 2:
        raise ValueError("DINO requires at least two global views")

    student_global: list[torch.Tensor] = []
    student_cls: list[torch.Tensor] = []
    student_global_views: list[torch.Tensor] = []
    for view in views:
        resized = _resize_for_backbone(view, model.student)
        cls, _ = model.student.forward_features(resized)
        student_global.append(model.student_head(cls))
        student_cls.append(cls)
        if len(student_global_views) < 2:
            student_global_views.append(resized)

    teacher_global: list[torch.Tensor] = []
    teacher_patches: list[torch.Tensor] = []
    with torch.no_grad():
        for view in views[:2]:
            resized = _resize_for_backbone(view, model.teacher)
            cls, patches = model.teacher.forward_features(resized)
            teacher_global.append(model.teacher_head(cls))
            teacher_patches.append(model.teacher_ibot_head(patches))

    paired_students: list[torch.Tensor] = []
    paired_teachers: list[torch.Tensor] = []
    for teacher_index, teacher_logits in enumerate(teacher_global):
        for student_index, student_logits in enumerate(student_global):
            if student_index == teacher_index:
                continue
            paired_students.append(student_logits)
            paired_teachers.append(teacher_logits)
    dino = model.dino_loss(
        torch.cat(paired_students), torch.cat(paired_teachers), teacher_temp=teacher_temp
    )

    ibot_students: list[torch.Tensor] = []
    ibot_teachers: list[torch.Tensor] = []
    for view, teacher_logits in zip(student_global_views, teacher_patches):
        mask = _patch_mask(
            teacher_logits.shape[0],
            teacher_logits.shape[1],
            mask_ratio,
            teacher_logits.device,
        )
        _, masked_patches = model.student.forward_features(view, mask=mask)
        student_logits = model.student_ibot_head(masked_patches)
        ibot_students.append(student_logits[mask])
        ibot_teachers.append(teacher_logits[mask])
    if ibot_students and sum(item.shape[0] for item in ibot_students):
        ibot = model.ibot_loss(
            torch.cat(ibot_students),
            torch.cat(ibot_teachers),
            teacher_temp=teacher_temp,
        )
    else:
        ibot = dino.new_zeros(())

    koleo = model.koleo_loss(student_cls[0])
    gram = gram_loss(
        student_cls[0],
        gram_enabled=spec.gram_enabled,
    )
    total = (
        dino
        + spec.ibot_weight * ibot
        + spec.koleo_weight * koleo
        + spec.gram_weight * gram
    )
    components = {
        "dino": dino.detach(),
        "ibot": ibot.detach(),
        "koleo": koleo.detach(),
        "gram": gram.detach(),
    }
    return total, components
