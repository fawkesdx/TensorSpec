"""Tests for suite-scoped live monitor log selection."""

from tensorspec.core.compute.cluster_live_monitor import _infer_job_from_processes


def test_infer_qe_from_run_pipeline():
    ps = "12345 600 Sl 99.0 2.1 bash run_pipeline.sh"
    assert _infer_job_from_processes(ps) == "qe"


def test_infer_chinook_from_remote_runner():
    ps = "999 120 Sl 80.0 3.0 python -u chinook_remote_runner.py --engine grizzly"
    assert _infer_job_from_processes(ps) == "chinook"


def test_infer_sprkkr_from_kkrscf():
    ps = "555 30 R 100.0 1.0 /path/kkrscf9.7 < sys.inp"
    assert _infer_job_from_processes(ps) == "sprkkr"


def test_infer_none_when_idle():
    assert _infer_job_from_processes("") is None
