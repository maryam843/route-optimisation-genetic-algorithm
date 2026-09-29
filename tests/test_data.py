import numpy as np
import pytest

from route_ga.data import MISSING_EDGE, is_symmetric, load_map, load_matrix, missing_mask


@pytest.mark.parametrize(("n", "missing"), [(32, 192), (100, 1900)])
def test_course_maps_shape_and_missing_edges(n, missing):
    d = load_map(n)
    assert d.shape == (n, n)
    assert int(missing_mask(d).sum()) == missing
    assert is_symmetric(d)
    assert np.all(np.diag(d) == 0)
    assert d[~missing_mask(d)].max() < 100


def test_threshold_is_100000_not_10000():
    assert MISSING_EDGE == 100_000
    assert missing_mask(np.array([[0, 100000.0], [99999.0, 250000.0]])).tolist() == [
        [False, True], [False, True]]


def test_unknown_map():
    with pytest.raises(ValueError):
        load_map(20)


def test_load_matrix_rejects_non_square(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("0,1,2\n0,1,2\n3,4,5\n")
    with pytest.raises(ValueError):
        load_matrix(p)
