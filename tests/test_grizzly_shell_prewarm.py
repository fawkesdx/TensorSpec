"""Ensure TensorSpec delegates reusable ME-shell setup to GrizzlyME."""

import re
from types import SimpleNamespace

import numpy as np


def test_build_grizzly_me_shell_import_prepare():
    from tensorspec.core.arpes.one_step import chinook_arpes_kmesh as k

    assert hasattr(k, "build_grizzly_me_shell")

    import inspect

    src = inspect.getsource(k.build_grizzly_me_shell)
    assert "prepare_me_shell" in src


def test_build_grizzly_me_shell_logs_radint_result(monkeypatch, capsys):
    import collections
    import collections.abc

    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable

    import chinook.ARPES_lib as arpes_lib
    import grizzly
    from tensorspec.core.arpes.one_step import chinook_arpes_kmesh as k

    exp = SimpleNamespace(
        basis=["orbital"],
        cube=((0.0, 0.0, 1), (0.0, 0.0, 1), (-1.0, 1.0, 2)),
        hv=21.2,
        W=4.5,
    )
    source = {"value": "torch"}

    def prepare_me_shell(_exp):
        return SimpleNamespace(
            basis=exp.basis,
            prefactors=np.ones(1),
            Largs=[],
            Margs=[],
            Gbasis=np.ones((1, 1)),
            proj_arr=np.ones(1),
            Bfuncs=["B"],
            radint_pointers=np.array([0]),
            nstates=1,
            spin=False,
            mfp=10.0,
            radint_source=source["value"],
        )

    monkeypatch.setattr(k, "apply_chinook_runtime_patches", lambda: None)
    monkeypatch.setattr(k, "build_arpes_dict_from_physics", lambda *a, **kw: {})
    monkeypatch.setattr(arpes_lib, "experiment", lambda *a, **kw: exp)
    monkeypatch.setattr(
        arpes_lib,
        "all_Y",
        lambda basis: ([], [], np.ones((1, 1)), np.array([0])),
    )
    monkeypatch.setattr(grizzly, "prepare_me_shell", prepare_me_shell)

    kwargs = {
        "tb_model": object(),
        "physics": {"matrix_element_mode": "Full Matrix Elements"},
        "B_matrix": np.eye(3),
        "e_axis": np.array([-1.0, 1.0]),
        "A_bulk": np.eye(3),
    }
    def _radint_line(captured: str) -> str:
        for line in captured.splitlines():
            if line.startswith("Radint "):
                return line.strip()
        raise AssertionError(f"no Radint log line in:\n{captured}")

    k.build_grizzly_me_shell(**kwargs)
    miss_line = _radint_line(capsys.readouterr().out)
    assert re.fullmatch(
        r"Radint MISS backend=auto source=torch wall=\d+\.\d{2}s",
        miss_line,
    )

    source["value"] = "cache"
    k.build_grizzly_me_shell(**kwargs)
    hit_line = _radint_line(capsys.readouterr().out)
    assert re.fullmatch(
        r"Radint HIT backend=auto source=cache wall=\d+\.\d{2}s",
        hit_line,
    )
