import torch

from tensorspec.core.ml.ssl.dino import (
    DINOHead,
    DinoLoss,
    DinoModel,
    KoLeoLoss,
    _patch_mask,
    dino_total_loss,
    gram_loss,
    update_teacher,
)
from tensorspec.core.ml.ssl.models.vit2d import build_vit2d
from tensorspec.core.ml.ssl.spec import DinoSpec, ModelSpec


def _tiny_dino():
    spec = DinoSpec(
        out_dim=64,
        hidden_dim=64,
        bottleneck_dim=32,
        ibot_weight=1.0,
        koleo_weight=0.1,
    )
    student = build_vit2d(ModelSpec(name="vit_ti", img_size=32, patch_size=8))
    teacher = build_vit2d(ModelSpec(name="vit_ti", img_size=32, patch_size=8))
    teacher.load_state_dict(student.state_dict())
    return DinoModel(student, teacher, spec), spec


def test_one_step_decreases_loss():
    torch.manual_seed(0)
    model, spec = _tiny_dino()
    opt = torch.optim.AdamW(model.student_parameters(), lr=1e-3)
    views = [torch.randn(4, 1, 32, 32) for _ in range(2)]
    model.train()
    loss0, _ = dino_total_loss(
        model, views, spec, teacher_temp=0.04, mask_ratio=0.3
    )
    loss0_value = loss0.item()
    loss0.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    update_teacher(model, momentum=0.9)
    loss1, _ = dino_total_loss(
        model, views, spec, teacher_temp=0.04, mask_ratio=0.3
    )
    assert loss1.item() < loss0_value


def test_ema_teacher_moves():
    model, _ = _tiny_dino()
    before = next(model.teacher.parameters()).detach().clone()
    with torch.no_grad():
        for parameter in model.student.parameters():
            parameter.add_(0.1)
    update_teacher(model, momentum=0.5)
    after = next(model.teacher.parameters()).detach()
    assert not torch.allclose(before, after)


def test_heads_losses_and_center_use_float32():
    spec = DinoSpec(out_dim=8, hidden_dim=16, bottleneck_dim=4)
    head = DINOHead(6, spec)
    logits = head(torch.randn(3, 6))
    assert logits.shape == (3, 8)

    loss_fn = DinoLoss(spec)
    loss = loss_fn(logits.to(torch.bfloat16), logits.detach().to(torch.bfloat16))
    assert loss.dtype == torch.float32
    assert loss_fn.center.dtype == torch.float32


def test_koleo_is_finite_and_gram_defaults_off():
    features = torch.randn(4, 12)
    loss = KoLeoLoss()(features)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert gram_loss(features, gram_enabled=False).item() == 0.0


def test_ibot_mask_token_receives_backbone_gradient():
    torch.manual_seed(7)
    model, spec = _tiny_dino()
    views = [torch.randn(3, 1, 32, 32) for _ in range(2)]
    loss, _ = dino_total_loss(model, views, spec, mask_ratio=0.5)
    loss.backward()
    gradient = model.student.mask_token.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0


def test_ibot_masks_random_positions_per_sample():
    torch.manual_seed(11)
    mask = _patch_mask(4, 16, 0.5, torch.device("cpu"))
    assert mask.sum(dim=1).tolist() == [8, 8, 8, 8]
    assert len({tuple(row.tolist()) for row in mask}) > 1


def test_configured_global_crop_count_controls_teacher_views():
    torch.manual_seed(13)
    model, spec = _tiny_dino()
    teacher_calls = 0
    original_forward = model.teacher.forward_features

    def count_teacher_views(*args, **kwargs):
        nonlocal teacher_calls
        teacher_calls += 1
        return original_forward(*args, **kwargs)

    model.teacher.forward_features = count_teacher_views
    views = [torch.randn(2, 1, 32, 32) for _ in range(4)]

    loss, _ = dino_total_loss(model, views, spec, n_global=3, mask_ratio=0.5)

    assert torch.isfinite(loss)
    assert teacher_calls == 3
