import numpy as np
from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import compute_A_lab
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
