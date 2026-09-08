import numpy as np
import torch

from tensorspec.core.ml.ssl.dino import DinoModel
from tensorspec.core.ml.ssl.models.vit2d import build_vit2d
from tensorspec.core.ml.ssl.probe import (
    ProbeConfig,
    agreement_metrics,
    cluster_embeddings,
    extract_cls_embeddings,
    filter_manifest_indices,
    labels_to_grid,
    load_dino_for_probe,
    probe,
    reference_to_binary,
    spatial_contiguity,
)
from tensorspec.core.ml.ssl.spec import (
    AugmentSpec,
    DinoSpec,
    ModelSpec,
    OptimSpec,
    RunConfig,
    to_jsonable,
)


def _tiny_ckpt(tmp_path):
    cfg = RunConfig(
        augment=AugmentSpec(arm="A1", n_global=2, n_local=0, global_size=32, local_size=32),
        model=ModelSpec(name="vit_ti", img_size=32, patch_size=8, in_chans=1),
        dino=DinoSpec(out_dim=64, hidden_dim=64, bottleneck_dim=32),
        optim=OptimSpec(epochs=1, batch_size=2, lr=1e-3),
        seed=0,
    )
    student = build_vit2d(cfg.model)
    teacher = build_vit2d(cfg.model)
    model = DinoModel(student, teacher, cfg.dino)
    path = tmp_path / "last.pt"
    torch.save(
        {
            "step": 1,
            "epoch": 0,
            "model": model.state_dict(),
            "optimizer": {},
            "scaler": {},
            "config": to_jsonable(cfg),
            "seed": 0,
            "git_commit": "test",
        },
        path,
    )
    return path, cfg


def test_filter_manifest_indices_by_source():
    manifest = {
        "samples": [
            {"source_id": "a.h5", "shard": 0, "offset": 0, "index": {"y": 0, "x": 0}},
            {"source_id": "b.h5", "shard": 0, "offset": 1, "index": {"y": 0, "x": 1}},
            {"source_id": "a.h5", "shard": 0, "offset": 2, "index": {"y": 1, "x": 0}},
        ]
    }
    assert filter_manifest_indices(manifest, "a.h5") == [0, 2]
    assert filter_manifest_indices(manifest, "missing.h5") == []


def test_probe_rejects_k_not_2(tmp_path):
    import pytest

    with pytest.raises(ValueError, match=r"k=2"):
        probe(
            ckpt=tmp_path / "missing.pt",
            data_dir=tmp_path,
            reference=tmp_path / "missing.npz",
            out_dir=tmp_path / "out",
            config=ProbeConfig(source_id="x", k=3),
        )


def test_extract_cls_shape(tmp_path):
    path, cfg = _tiny_ckpt(tmp_path)
    model, loaded_cfg = load_dino_for_probe(path, device=torch.device("cpu"))
    images = np.random.randn(5, 32, 32).astype(np.float32)
    emb = extract_cls_embeddings(
        model, images, batch_size=2, use_teacher=True, device=torch.device("cpu")
    )
    assert emb.shape == (5, model.teacher.embed_dim)


def test_cluster_two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, size=(40, 8))
    b = rng.normal(5, 0.1, size=(40, 8))
    emb = np.vstack([a, b]).astype(np.float32)
    labels = cluster_embeddings(emb, ProbeConfig(source_id="x", k=2, pca_dim=4, seed=0))
    assert set(labels.tolist()) == {0, 1}
    assert labels[:40].mean() != labels[40:].mean()  # separated



def test_labels_to_grid_fills_yx():
    assigns = np.array([0, 1, 0, 1])
    prov = [
        {"index": {"y": 0, "x": 0}},
        {"index": {"y": 0, "x": 1}},
        {"index": {"y": 1, "x": 0}},
        {"index": {"y": 1, "x": 1}},
    ]
    grid = labels_to_grid(assigns, prov, ny=2, nx=2)
    np.testing.assert_array_equal(grid, [[0, 1], [0, 1]])


def test_labels_to_grid_rejects_hole():
    import pytest
    with pytest.raises(ValueError, match="incomplete"):
        labels_to_grid(
            np.array([0]),
            [{"index": {"y": 0, "x": 0}}],
            ny=2,
            nx=2,
        )


def test_agreement_permutation_invariant():
    ref = np.array([[0, 0], [1, 1]])
    pred = np.array([[1, 1], [0, 0]])  # swapped labels
    m = agreement_metrics(pred, ref)
    assert m["ari"] == 1.0
    assert m["nmi"] == 1.0
    assert m["iou"] == 1.0


def test_contiguity_perfect_blocks():
    lab = np.array([[0, 0], [1, 1]])
    assert spatial_contiguity(lab) == 1.0


def test_reference_to_binary_two_level_map():
    low = np.full((3, 4), 1.0, dtype=np.float32)
    high = np.full((3, 4), 10.0, dtype=np.float32)
    map_ = np.hstack([low, high])
    labels = reference_to_binary(map_, seed=0)
    assert labels.shape == map_.shape
    assert set(np.unique(labels).tolist()) == {0, 1}
    left = np.unique(labels[:, :4])
    right = np.unique(labels[:, 4:])
    assert left.size == 1 and right.size == 1
    assert left[0] != right[0]
