"""Ensure TensorSpec delegates reusable ME-shell setup to GrizzlyME."""


def test_build_grizzly_me_shell_import_prepare():
    from tensorspec.core.arpes.one_step import chinook_arpes_kmesh as k

    assert hasattr(k, "build_grizzly_me_shell")

    import inspect

    src = inspect.getsource(k.build_grizzly_me_shell)
    assert "prepare_me_shell" in src
