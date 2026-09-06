# tests/test_ssl_augment.py
import numpy as np
import pytest
import torch

from tensorspec.core.ml.ssl.augment import MultiCropDataset, apply_view, build_multi_crop
from tensorspec.core.ml.ssl.spec import AugmentSpec


def test_a1_preserves_shape_and_finiteness():
    img = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
    spec = AugmentSpec(arm="A1", n_global=2, n_local=6)
    rng = np.random.default_rng(0)
    views = build_multi_crop(img, spec, rng)
    assert len(views) == 8
    assert views[0].shape == (spec.global_size, spec.global_size)
    assert views[2].shape == (spec.local_size, spec.local_size)
    assert all(np.isfinite(v).all() for v in views)


def test_a0_mirror_flips_chirality_marker():
    """Left-half bright / right-half dark must swap under forced mirror."""
    img = np.zeros((128, 128), dtype=np.float32)
    img[:, :64] = 1.0
    spec = AugmentSpec(arm="A0", mirror_prob=1.0, n_global=1, n_local=0)
    # Bypass multi-crop: call internal mirror via apply_view with fixed rng
    out = apply_view(img, spec, rng=np.random.default_rng(0), kind="global")
    assert out.shape[0] == spec.global_size
    # After mirror + crop/resize, left should be darker than right on average
    # Use pure mirror helper if exposed:
    from tensorspec.core.ml.ssl.augment import mirror_flip

    flipped = mirror_flip(img)
    assert flipped[:, :64].mean() < flipped[:, 64:].mean()
    assert abs(flipped[:, ::-1] - img).max() < 1e-6


def test_a1_has_no_mirror_effect_on_marker():
    img = np.zeros((128, 128), dtype=np.float32)
    img[:, :64] = 1.0
    from tensorspec.core.ml.ssl.augment import mirror_flip

    # A1 path must not call mirror — compare mean left/right after many views
    spec = AugmentSpec(arm="A1", n_global=4, n_local=0, global_crop_frac=1.0)
    rng = np.random.default_rng(1)
    views = build_multi_crop(img, spec, rng)
    for v in views:
        # full-frame crop: left still brighter
        assert v[:, : v.shape[1] // 2].mean() > v[:, v.shape[1] // 2 :].mean()


def test_multi_crop_dataset_changes_rng_each_epoch_but_remains_reproducible():
    image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    base = [(image, {})]
    spec = AugmentSpec(
        arm="A1",
        n_global=2,
        n_local=0,
        global_size=24,
        global_crop_frac=0.75,
    )
    dataset = MultiCropDataset(base, spec, seed=17)

    dataset.set_epoch(0)
    epoch_zero = dataset[0]
    dataset.set_epoch(1)
    epoch_one = dataset[0]
    dataset.set_epoch(0)
    epoch_zero_again = dataset[0]

    assert epoch_zero.valid and epoch_one.valid
    assert not torch.equal(epoch_zero[0], epoch_one[0])
    assert torch.equal(epoch_zero[0], epoch_zero_again[0])


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((16, 16), dtype=np.float32),
        np.full((16, 16), np.nan, dtype=np.float32),
    ],
)
def test_multi_crop_dataset_flags_invalid_source_samples(image):
    dataset = MultiCropDataset(
        [(image, {})],
        AugmentSpec(arm="A1", n_global=2, n_local=0, global_size=16),
    )

    views = dataset[0]

    assert not views.valid
