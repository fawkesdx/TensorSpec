import numpy as np
from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import (
    compute_A_lab,
    physics_from_experiment_kwargs,
)


def test_A_lab_lh_differs_55_vs_20():
    a55 = compute_A_lab("Linear Horizontal (p-pol)", 55.0)
    a20 = compute_A_lab("Linear Horizontal (p-pol)", 20.0)
    assert not np.allclose(a55, a20)
    assert np.allclose(a55, [np.cos(np.radians(55)), -np.sin(np.radians(55)), 0.0])


def test_physics_maps_rad_type_mfp_kz():
    phys = physics_from_experiment_kwargs(
        {
            "photon_energy": 84.0,
            "rad_type": "hydrogenic",
            "mfp": 5.0,
            "kz_halfwidth": 0.2,
            "kz_npoints": 7,
        }
    )
    assert phys["rad_type"] == "hydrogenic"
    assert phys["mfp"] == 5.0
    assert phys["kz_halfwidth"] == 0.2
    assert phys["kz_npoints"] == 7


def test_physics_json_roundtrip_keys():
    raw = {
        "photon_energy": 84.0,
        "rad_type": "hydrogenic",
        "mfp": 5.0,
        "kz_halfwidth": 0.2,
        "kz_npoints": 7,
        "incidence_angle": 20.0,
    }
    phys = physics_from_experiment_kwargs(raw)
    assert phys["incidence_angle"] == 20.0
    assert phys["rad_type"] == "hydrogenic"


def test_physics_defaults_preserve_legacy():
    phys = physics_from_experiment_kwargs({"photon_energy": 84.0})
    assert phys["rad_type"] == "slater"
    assert phys["mfp"] == 10.0
    assert phys["kz_halfwidth"] == 0.0
    assert phys["kz_npoints"] == 1


def test_physics_kz_npoints_defaults_when_halfwidth_on():
    phys = physics_from_experiment_kwargs(
        {"photon_energy": 84.0, "kz_halfwidth": 0.2}
    )
    assert phys["kz_halfwidth"] == 0.2
    assert phys["kz_npoints"] == 7


def test_arpes_dict_uses_Vo_and_rad_type():
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import build_arpes_dict_from_physics

    d = build_arpes_dict_from_physics(
        {
            "hv": 84.0,
            "work_function": 4.5,
            "inner_potential": 12.0,
            "rad_type": "hydrogenic",
            "mfp": 5.0,
        },
        cube={"X": [0, 0, 1], "Y": [0, 0, 1], "E": [-1, 0, 2]},
        energy_axis=np.linspace(-1, 0, 2),
        A_bulk=np.array([1.0, 0.0, 0.0]),
        is_full=True,
    )
    assert "Vo" in d and "V0" not in d
    assert d["Vo"] == 12.0
    assert d["rad_type"] == "hydrogenic"
    assert d["mfp"] == 5.0


def test_arpes_dict_preserves_circular_pol():
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import (
        build_arpes_dict_from_physics,
        sample_to_bulk_frame,
    )

    a_cr = compute_A_lab("Circular Right (CR)", 55.0)
    a_cl = compute_A_lab("Circular Left (CL)", 55.0)
    assert not np.allclose(a_cr, a_cl)
    assert np.iscomplexobj(a_cr) and np.any(np.abs(np.imag(a_cr)) > 1e-10)

    A = sample_to_bulk_frame(a_cr, 0.0, 0.0, 0.0)
    d = build_arpes_dict_from_physics(
        {"hv": 84.0, "work_function": 4.5, "inner_potential": 12.0},
        cube={"X": [0, 0, 1], "Y": [0, 0, 1], "E": [-1, 0, 2]},
        energy_axis=np.linspace(-1, 0, 2),
        A_bulk=A,
        is_full=True,
    )
    assert np.iscomplexobj(d["pol"])
    assert np.allclose(d["pol"], A)
    assert np.any(np.abs(np.imag(d["pol"])) > 1e-10)


def test_kz_weights_legacy_off():
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import lorentzian_kz_weights

    dk, w = lorentzian_kz_weights(0.0, 7)
    assert list(dk) == [0.0]
    assert list(w) == [1.0]


def test_kz_weights_normalized():
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import lorentzian_kz_weights

    dk, w = lorentzian_kz_weights(0.2, 7)
    assert len(dk) == 7
    assert np.isclose(w.sum(), 1.0)
    assert dk[0] < 0 < dk[-1]


def test_kz_weights_npoints_one_with_halfwidth_uses_seven():
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import lorentzian_kz_weights

    dk, w = lorentzian_kz_weights(0.2, 1)
    assert len(dk) == 7
    assert np.isclose(w.sum(), 1.0)


def test_kz_shift_along_surface_normal_not_cartesian_z():
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import (
        get_hkl_surface_frame,
        shift_k_bulk_along_surface_normal,
    )

    B = np.eye(3)
    hkl = (-2, 0, 1)
    Z_surf, _ = get_hkl_surface_frame(hkl, B, azimuthal_ref=np.array([0.0, 1.0, 0.0]))
    assert abs(Z_surf[0]) > 1e-6  # tilted: not || bulk ẑ

    K0 = np.zeros((3, 5))
    dk = 0.2
    K1 = shift_k_bulk_along_surface_normal(K0, dk, hkl, B)
    delta = K1 - K0
    assert np.allclose(delta, dk * Z_surf[:, None])
    assert np.max(np.abs(delta[0])) > 1e-6  # nonzero non-z cartesian component
