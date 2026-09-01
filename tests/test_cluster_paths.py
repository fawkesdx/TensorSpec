"""Unit tests for remote cluster path resolution."""

from tensorspec.core.compute.cluster_paths import (
    build_qe_slurm_batch,
    heavy_root,
    is_slurm,
    job_dir,
    mpi_launch_prefix,
    python_bin,
    repo_root,
    slurm_account,
    slurm_qos,
    slurm_walltime,
    sprkkr_binary,
    tmp_dir,
)


def test_lbl_defaults_when_paths_omitted():
    c = {"user": "alice", "host": "h", "name": "n"}
    assert heavy_root(c) == "/mnt/data/alice/tensorspec_heavy"
    assert tmp_dir(c) == "/mnt/data/alice/tmp"
    assert repo_root(c) == "/home/alice/TensorSpec"
    assert python_bin(c) == "/home/alice/TensorSpec/TensorSpec_env/bin/python"
    assert job_dir(c, "chinook") == "/mnt/data/alice/tensorspec_heavy/chinook_gui_run"


def test_custom_paths_block():
    c = {
        "user": "alice",
        "paths": {
            "heavy_root": "/scratch/ts",
            "tmp_dir": "/scratch/tmp",
            "repo_root": "/projects/ts",
            "python": "/projects/ts/.venv/bin/python",
            "sprkkr_bin": "/opt/sprkkr/bin",
        },
    }
    assert heavy_root(c) == "/scratch/ts"
    assert tmp_dir(c) == "/scratch/tmp"
    assert python_bin(c) == "/projects/ts/.venv/bin/python"
    assert job_dir(c, "qe") == "/scratch/ts/qe_gui_run"
    assert sprkkr_binary(c, "kkrspec9.7") == "/opt/sprkkr/bin/kkrspec9.7"


def test_legacy_top_level_keys():
    c = {
        "user": "bob",
        "heavy_root": "/data/heavy",
        "python": "/usr/bin/python3",
    }
    assert heavy_root(c) == "/data/heavy"
    assert python_bin(c) == "/usr/bin/python3"


def test_slurm_helpers_and_batch():
    c = {
        "user": "alice",
        "mode": "SLURM",
        "paths": {
            "heavy_root": "/scratch/ts",
            "qe_module": "espresso/7.5-libxc-7.0.0-cpu",
            "slurm": {
                "account": "m9999",
                "qos": "regular_0",
                "constraint": "cpu",
                "walltime": "06:00:00",
            },
        },
    }
    assert is_slurm(c)
    assert slurm_account(c) == "m9999"
    assert slurm_qos(c) == "regular_0"
    assert slurm_walltime(c) == "06:00:00"
    assert mpi_launch_prefix(c, 128).startswith("srun -n 128")
    batch = build_qe_slurm_batch(
        c, remote_dir="/scratch/ts/qe_gui_run", script_name="run_pipeline.sh", mpi_ranks=128
    )
    assert "#SBATCH -A m9999" in batch
    assert "module load espresso/7.5-libxc-7.0.0-cpu" in batch
    assert "bash run_pipeline.sh" in batch
