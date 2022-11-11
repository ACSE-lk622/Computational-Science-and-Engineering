import numpy as np
import os
import pytest
import functions as f

BASE_PATH = os.path.dirname(__file__)


@pytest.mark.parametrize('input, output', [([[2, 0, -1], [0, 5, 6],
                                             [0, -1, 1]], 22),
                                           ([[2, 4, 5], [3, 5, 6],
                                             [2, 5, 2]], 9)])
def test_det(input, output):
    result = f.det(input)
    assert result == output


@pytest.mark.parametrize('input, input2, output', [([[1, 2], [3, 4]],
                                                    [[5], [6]],
                                                    [[17.], [39.]]),
                                                   ([[2, 4, 5], [3, 5, 6],
                                                     [2, 5, 2]],
                                                    [[1], [2], [-1]],
                                                    [[5.], [7.], [10.]])])
def test_mult(input, input2, output):
    result = f.mult(input, input2)
    assert (result == output).all()


@pytest.mark.parametrize('input, output', [([[1, 0, -1], [-2, 3, 0],
                                             [1, -3, 2]],
                                            [[6., 3., 3.], [4., 3., 2.],
                                             [3., 3., 3.]]),
                                           ([[2, 4, 5, 6], [3, 4, 6, 2],
                                             [2, 3, 2, 1], [2, 1, 4, 5]],
                                            [[-40., -14., 58., 42.],
                                             [22., -4., 11., -27.],
                                             [2., 28., -38., -6.],
                                             [10., -16., 5., 9.]])])
def test_adj(input, output):
    result = f.adj(input)
    assert np.isclose(result, output).all()


@pytest.mark.parametrize('input, output', [([[1, 0, -1], [-2, 3, 0],
                                             [1, -3, 2]],
                                            np.linalg.inv([[1, 0, -1],
                                                           [-2, 3, 0],
                                                           [1, -3, 2]])),
                                           ([[2, 0, 3], [0, 5, 5], [0, -1, 2]],
                                            np.linalg.inv([[2, 0, 3],
                                                           [0, 5, 5],
                                                           [0, -1, 2]]))])
def test_inv(input, output):
    result = f.inv(input)
    assert np.isclose(result, output).all()


@pytest.mark.parametrize('input, input2, output', [([[2, 0, 3], [0, 5, 5],
                                                     [0, -1, 2]], [[1], [2],
                                                                   [-1]],
                                                    np.linalg.solve(
                                                     [[2, 0, 3],
                                                      [0, 5, 5],
                                                      [0, -1, 2]],
                                                     [[1], [2], [-1]])),
                                                   ([[1, 0, -1], [-2, 3, 0],
                                                     [1, -3, 2]],
                                                    [[1], [2], [-1]],
                                                    np.linalg.solve(
                                                     [[1, 0, -1],
                                                      [-2, 3, 0],
                                                      [1, -3, 2]],
                                                     [[1], [2],
                                                      [-1]]))])
def test_solve(input, input2, output):
    result = f.solve(input, input2)
    assert np.isclose(result, output).all()


# @pytest.mark.parametrize('input, input2, output', [([[2, 0, 3], [0, 5, 5],
#                                                      [0, -1, 2]], [[1], [2],
#                                                                    [-1]],
#                                                     [[0.8], [0.6],
#                                                            [-0.2]]),
#                                                    ([[1, 0, -1], [-2, 3, 0],
#                                                      [1, -3, 2]],
#                                                     [[1], [2], [-1]],
#                                                     [[3.0],
#                                                     [2.66666667],
#                                                     [2.0]])])
# def test_new_solve(input, input2, output):
#     result = f.new_solve(input, input2)
#     assert np.isclose(result, output).all()
