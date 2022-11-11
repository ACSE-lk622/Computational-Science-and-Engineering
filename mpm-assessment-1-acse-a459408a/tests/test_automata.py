"""Test automata module functions."""

import os

import numpy as np

import automata  # noqa

BASE_PATH = os.path.dirname(__file__)


def test_lorenz96():
    """Test Lorenz 96 implementation"""
    initial64 = np.load(os.sep.join((BASE_PATH,
                                     'lorenz96_64_init.npy')))

    onestep64 = np.load(os.sep.join((BASE_PATH,
                                     'lorenz96_64_onestep.npy')))
    assert np.isclose(automata.lorenz96(initial64, 1), onestep64).all()

    thirtystep64 = np.load(os.sep.join((BASE_PATH,
                                        'lorenz96_64_thirtystep.npy')))
    assert np.isclose(automata.lorenz96(initial64, 30), thirtystep64).all()


def test_life():
    x = np.array([[True, False, True, True],
                  [True, True, False, True], [True, False, True, False]])
    y = automata.life(x, 1)
    result = np.array([[True, False, True, True],
                       [True, False, False, True], [True, False, True, False]])
    assert (result == y).all()


def test_lift_per():
    x = np.array([[True, False, False, False],
                  [True, True, False, False], [True, False, True, False]])
    y = automata.life_periodic(x, 2)
    result = np.array([[True, True, False, True],
                       [True, True, False, True], [True, True, False, True]])
    assert (result == y).all()


def test_life2colour():
    x = np.array([[-1, 0, 1, -1], [0, 1, 0, -1], [0, 1, 0, 0]])
    y = automata.life2colour(x, 2)
    result = np.array([[1, 1, 0, -1], [1, 0, 0, -1], [0, 1, 1, 0]])
    assert (result == y).all()


def test_lifepent():
    x = np.array([[1, 1, 0, 0], [1, 1, 1, 0],
                  [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]])
    y = automata.lifepent(x, 2)
    result = np.array([[[0, 0, 0, 0], [0, 0, 1, 0],
                       [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]]])
    assert (result == y).all()

# This is the end of the five test.