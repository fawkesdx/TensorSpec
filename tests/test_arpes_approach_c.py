import numpy as np
from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import compute_A_lab
from tensorspec.core.arpes.one_step.fresnel import apply_fresnel_to_A_lab


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
