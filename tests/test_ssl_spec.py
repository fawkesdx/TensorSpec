"""Tests for SSL preprocess config dataclasses."""

from tensorspec.core.ml.ssl.spec import (
    NormSpec,
    PreprocessConfig,
    ResampleSpec,
    SampleSpec,
    TrimSpec,
    preprocess_config_from_dict,
    to_jsonable,
)


def test_preprocess_config_roundtrip():
    cfg = PreprocessConfig(
        trim=TrimSpec(ranges={"slit": (-10.0, 10.0)}, source_kind="xy_fine_4d"),
        norm=NormSpec(),
        resample=ResampleSpec(),
        sample=SampleSpec(mode="disp2d", index_roles=("y", "x")),
    )
    d = to_jsonable(cfg)
    cfg2 = preprocess_config_from_dict(d)
    assert cfg2.sample.mode == "disp2d"
    assert cfg2.trim.ranges["slit"] == (-10.0, 10.0)
