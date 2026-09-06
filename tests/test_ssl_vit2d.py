import torch

from tensorspec.core.ml.ssl.models.vit2d import build_vit2d
from tensorspec.core.ml.ssl.spec import ModelSpec


def test_vit_s_forward_features_shapes():
    model = build_vit2d(ModelSpec(name="vit_s", img_size=128, patch_size=16, in_chans=1))
    x = torch.randn(2, 1, 128, 128)
    cls, patches = model.forward_features(x)
    assert cls.shape == (2, 384)
    assert patches.shape == (2, 64, 384)  # 8*8 patches


def test_vit_ti_param_count_sanity():
    model = build_vit2d(ModelSpec(name="vit_ti"))
    n = sum(p.numel() for p in model.parameters())
    assert 4_000_000 < n < 8_000_000
