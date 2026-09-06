"""Multi-crop augmentations for SSL Stage-1 (arms A0 / A1)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom

from tensorspec.core.ml.ssl.spec import AugmentSpec

ViewKind = Literal["global", "local"]


def mirror_flip(image: np.ndarray) -> np.ndarray:
    """Horizontal flip along slit axis (axis 1)."""
    return np.ascontiguousarray(image[:, ::-1])


def _clamp01(image: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _resize_bilinear(image: np.ndarray, out_size: int) -> np.ndarray:
    h, w = image.shape
    if h == out_size and w == out_size:
        return image.astype(np.float32, copy=False)
    factors = (out_size / h, out_size / w)
    return zoom(image, factors, order=1).astype(np.float32)


def _translation_crop(
    image: np.ndarray,
    crop_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Translation-only crop at fixed scale fraction of the frame."""
    h, w = image.shape
    crop_frac = float(np.clip(crop_frac, 0.0, 1.0))
    crop_h = max(1, int(round(h * crop_frac)))
    crop_w = max(1, int(round(w * crop_frac)))
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)
    y0 = 0 if crop_h == h else int(rng.integers(0, h - crop_h + 1))
    x0 = 0 if crop_w == w else int(rng.integers(0, w - crop_w + 1))
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _random_resized_crop(
    image: np.ndarray,
    out_size: int,
    scale_min: float,
    scale_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = image.shape
    scale = float(rng.uniform(scale_min, scale_max))
    side = max(1, int(round(min(h, w) * scale)))
    side = min(side, h, w)
    y0 = 0 if side == h else int(rng.integers(0, h - side + 1))
    x0 = 0 if side == w else int(rng.integers(0, w - side + 1))
    cropped = image[y0 : y0 + side, x0 : x0 + side]
    return _resize_bilinear(cropped, out_size)


def _apply_poisson_noise(
    image: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if scale <= 0.0:
        return image
    counts = np.clip(image * scale, 0.0, None)
    noisy = rng.poisson(counts).astype(np.float32)
    return noisy / scale


def _apply_gain_jitter(
    image: np.ndarray,
    gain_jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if gain_jitter <= 0.0:
        return image
    gain = 1.0 + float(rng.uniform(-gain_jitter, gain_jitter))
    return image * gain


def _apply_per_edc_scale(
    image: np.ndarray,
    gain_jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if gain_jitter <= 0.0:
        return image
    out = image.copy()
    scales = 1.0 + rng.uniform(-gain_jitter, gain_jitter, size=out.shape[0])
    out *= scales[:, np.newaxis]
    return out


def _apply_energy_shift(
    image: np.ndarray,
    max_shift_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_shift_px <= 0.0:
        return image
    max_px = int(max_shift_px)
    if max_px == 0:
        return image
    shift = int(rng.integers(-max_px, max_px + 1))
    if shift == 0:
        return image
    return np.roll(image, shift=shift, axis=0)


def _apply_dead_pixels(
    image: np.ndarray,
    dead_pixel_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if dead_pixel_prob <= 0.0:
        return image
    mask = rng.random(image.shape) < dead_pixel_prob
    if not mask.any():
        return image
    out = image.copy()
    out[mask] = 0.0
    return out


def _apply_smooth_background(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = image.shape
    coarse_h = max(2, h // 16)
    coarse_w = max(2, w // 16)
    background = rng.uniform(0.0, 0.05, size=(coarse_h, coarse_w)).astype(np.float32)
    upsampled = zoom(background, (h / coarse_h, w / coarse_w), order=1)
    upsampled = upsampled[:h, :w]
    return image + upsampled


def _apply_smoothing(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return image
    return gaussian_filter(image, sigma=sigma).astype(np.float32, copy=False)


def _apply_safe_transforms(
    image: np.ndarray,
    spec: AugmentSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    out = image.astype(np.float32, copy=False)
    out = _apply_poisson_noise(out, spec.poisson_scale, rng)
    out = _apply_gain_jitter(out, spec.gain_jitter, rng)
    out = _apply_per_edc_scale(out, spec.gain_jitter, rng)
    out = _apply_energy_shift(out, spec.energy_shift_px, rng)
    out = _apply_dead_pixels(out, spec.dead_pixel_prob, rng)
    out = _apply_smooth_background(out, rng)
    out = _apply_smoothing(out, spec.smooth_sigma)
    return _clamp01(out)


def _maybe_mirror(
    image: np.ndarray,
    mirror_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if mirror_prob <= 0.0:
        return image
    if rng.random() < mirror_prob:
        return mirror_flip(image)
    return image


def apply_view(
    image: np.ndarray,
    spec: AugmentSpec,
    *,
    rng: np.random.Generator,
    kind: ViewKind,
) -> np.ndarray:
    """Build one augmented global or local view as float32 [H, W] in ~[0, 1]."""
    img = np.asarray(image, dtype=np.float32)
    out_size = spec.global_size if kind == "global" else spec.local_size
    crop_frac = spec.global_crop_frac if kind == "global" else spec.local_crop_frac

    if spec.arm == "A0":
        cropped = _maybe_mirror(img, spec.mirror_prob, rng)
        view = _random_resized_crop(
            cropped,
            out_size,
            spec.rrc_scale_min,
            spec.rrc_scale_max,
            rng,
        )
        return _clamp01(view)

    transformed = _apply_safe_transforms(img, spec, rng)
    if not np.any(transformed):
        cropped = _translation_crop(img, crop_frac, rng)
    else:
        cropped = _translation_crop(transformed, crop_frac, rng)
    view = _resize_bilinear(cropped, out_size)
    return _clamp01(view)


def build_multi_crop(
    image: np.ndarray,
    spec: AugmentSpec,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Return n_global + n_local augmented views."""
    views: list[np.ndarray] = []
    for _ in range(spec.n_global):
        views.append(apply_view(image, spec, rng=rng, kind="global"))
    for _ in range(spec.n_local):
        views.append(apply_view(image, spec, rng=rng, kind="local"))
    return views


class MultiCropViews(list[torch.Tensor]):
    """List-compatible crop result carrying source-sample validity."""

    def __init__(self, views: list[torch.Tensor], *, valid: bool) -> None:
        super().__init__(views)
        self.valid = valid


class MultiCropDataset(torch.utils.data.Dataset):
    def __init__(self, base, spec: AugmentSpec, seed: int = 0):
        self.base = base
        self.spec = spec
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        sample, _meta = self.base[index]
        img = np.asarray(sample, dtype=np.float32)
        valid = bool(np.isfinite(img).all() and np.any(img))
        if not valid:
            sizes = [self.spec.global_size] * self.spec.n_global + [
                self.spec.local_size
            ] * self.spec.n_local
            return MultiCropViews(
                [
                    torch.zeros((1, size, size), dtype=torch.float32)
                    for size in sizes
                ],
                valid=False,
            )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(index), self.epoch])
        )
        views = build_multi_crop(img, self.spec, rng)
        return MultiCropViews(
            [torch.from_numpy(v)[None, ...] for v in views],
            valid=True,
        )
