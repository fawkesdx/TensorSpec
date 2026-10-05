import math
import numpy as np
import pytest

from tensorspec.core.bernevig_twist import (
    classify_bravais,
    hexagonal_pair,
    q_lattice_label,
    square_pair,
)


def _hex_ab(a=2.46):
    return np.array([[a, 0.0], [-a / 2, a * math.sqrt(3) / 2]])


def test_bravais_and_q_labels():
    assert classify_bravais(_hex_ab()) == "hexagonal"
    assert classify_bravais(np.array([[3.0, 0.0], [0.0, 3.0]])) == "square"
    assert classify_bravais(np.array([[3.0, 0.0], [0.0, 5.0]])) == "rectangular"
    assert classify_bravais(np.array([[3.0, 0.0], [1.0, 4.0]])) == "oblique"
    assert q_lattice_label("hexagonal", "Γ") == "triangular"
    assert q_lattice_label("hexagonal", "K") == "honeycomb"
    assert q_lattice_label("hexagonal", "M") == "kagome"
    assert q_lattice_label("square", "X") == "nested square"
    assert q_lattice_label("square", "Γ") == "square"
    assert q_lattice_label("rectangular", "HSP") == "rectangular"
    with pytest.raises(ValueError):
        q_lattice_label("hexagonal", "X")


def test_hex_and_square_pairs_match_paper_angles():
    th, S = hexagonal_pair(5, 4)
    assert th == pytest.approx(7.340993016630217)
    assert abs(int(round(np.linalg.det(S)))) == 61
    assert S.tolist() == [[5, -4], [4, 9]]
    th98, S98 = hexagonal_pair(9, 8)
    assert th98 == pytest.approx(3.8902381690076835)
    assert abs(int(round(np.linalg.det(S98)))) == 217
    ths, Ss = square_pair(2, 1)
    assert ths == pytest.approx(53.13010235415598)
    assert abs(int(round(np.linalg.det(Ss)))) == 5
    assert Ss.tolist() == [[2, 1], [-1, 2]]
