import numpy as np
from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import (
    build_k_bulk_mesh,
    compute_A_lab,
    physics_from_experiment_kwargs,
)
from tensorspec.core.arpes.one_step.fresnel import apply_fresnel_to_A_lab
from tensorspec.core.arpes.one_step.photon_momentum import photon_q_lab
from tensorspec.core.kinematics import ARPESKinematics


def test_fresnel_n1_matches_vacuum_LH():
    for alpha in (20.0, 55.0):
        A = compute_A_lab("Linear Horizontal (p-pol)", alpha)
        Af = apply_fresnel_to_A_lab(A, alpha, n=1.0 + 0.0j)
        np.testing.assert_allclose(Af, A, atol=1e-12)


def test_fresnel_n1_matches_vacuum_LV():
    A = compute_A_lab("Linear Vertical (s-pol)", 55.0)
    Af = apply_fresnel_to_A_lab(A, 55.0, n=1.0)
    np.testing.assert_allclose(Af, A, atol=1e-12)


def test_fresnel_n_gt1_changes_LH_Ez_ratio():
    """With n>1, transmitted p-field mix must differ from vacuum at fixed α."""
    alpha = 55.0
    A = compute_A_lab("Linear Horizontal (p-pol)", alpha)
    Af = apply_fresnel_to_A_lab(A, alpha, n=2.0)
    assert not np.allclose(Af, A, atol=1e-6)


def _q_parallel_mag(q: np.ndarray) -> float:
    """|q∥| in lab frame: surface normal = +y → plane is x–z."""
    return float(np.hypot(q[0], q[2]))


def test_photon_q_magnitude():
    q = photon_q_lab(84.0, 55.0)
    assert abs(np.linalg.norm(q) - 84.0 / ARPESKinematics.HBAR_C) < 1e-10


def test_photon_q_parallel_delta_55_vs_20():
    """Critic golden number at 84 eV: Δ|q∥| ≈ 0.0203 Å⁻¹ between 55° and 20°."""
    q55 = photon_q_lab(84.0, 55.0)
    q20 = photon_q_lab(84.0, 20.0)
    dq = abs(_q_parallel_mag(q55) - _q_parallel_mag(q20))
    assert abs(dq - 0.0203) < 5e-4


def test_physics_approach_c_defaults():
    phys = physics_from_experiment_kwargs({"photon_energy": 84.0})
    assert phys["fresnel_enabled"] is True
    assert phys["optical_n"] == 1.0
    assert phys["optical_k"] == 0.0
    assert phys["include_photon_momentum"] is False


def _tiny_mesh_kwargs():
    return dict(
        k_bounds={
            "X": [-5.0, 5.0, 3],
            "Y": [0.0, 0.0, 1],
            "E": [-1.0, 0.0, 2],
        },
        hv=84.0,
        work_function=4.5,
        inner_potential=12.0,
        slit_angle=0.0,
        manip_theta=0.0,
        manip_azimuth=0.0,
        manip_tilt=0.0,
        incidence_angle=55.0,
        polarization="Linear Horizontal (p-pol)",
        hkl=(0, 0, 1),
        B_matrix=np.eye(3),
        lin_pol_angle=45.0,
    )


def test_build_k_mesh_photon_q_off_matches_baseline():
    """Tiny synthetic mesh: photon-q off + Fresnel n=1 == vacuum/legacy path."""
    kw = _tiny_mesh_kwargs()
    bounds = kw.pop("k_bounds")
    K_legacy, A_legacy, *_ = build_k_bulk_mesh(
        bounds,
        **kw,
        fresnel_enabled=False,
        include_photon_momentum=False,
    )
    K_c, A_c, *_ = build_k_bulk_mesh(
        bounds,
        **kw,
        fresnel_enabled=True,
        optical_n=1.0,
        optical_k=0.0,
        include_photon_momentum=False,
    )
    np.testing.assert_allclose(K_c, K_legacy, atol=1e-12)
    np.testing.assert_allclose(A_c, A_legacy, atol=1e-12)


def test_build_k_mesh_photon_q_on_shifts_k():
    """Photon-q on subtracts rotated q from K_BULK; A unchanged."""
    kw = _tiny_mesh_kwargs()
    bounds = kw.pop("k_bounds")
    K_off, A_off, *_ = build_k_bulk_mesh(
        bounds,
        **kw,
        fresnel_enabled=False,
        include_photon_momentum=False,
    )
    K_on, A_on, *_ = build_k_bulk_mesh(
        bounds,
        **kw,
        fresnel_enabled=False,
        include_photon_momentum=True,
    )
    np.testing.assert_allclose(A_on, A_off, atol=1e-12)
    assert not np.allclose(K_on, K_off, atol=1e-10)
    delta = K_off - K_on
    for i in range(delta.shape[1]):
        np.testing.assert_allclose(delta[:, i], delta[:, 0], atol=1e-12)
    q_lab = photon_q_lab(kw["hv"], kw["incidence_angle"])
    assert abs(np.linalg.norm(delta[:, 0]) - np.linalg.norm(q_lab)) < 1e-10
