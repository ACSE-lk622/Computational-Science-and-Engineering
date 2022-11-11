"""Implementations of Lorenz 96 and Conway's
Game of Life on various meshes"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def lorenz96(initial_state, nsteps):
    """
    Perform iterations of the Lorenz 96 update.

    Parameters
    ----------
    initial_state : array_like or list
        Initial state of lattice in an array of floats.
    nsteps : int
        Number of steps of Lorenz 96 to perform.

    Returns
    -------

    numpy.ndarray
         Final state of lattice in array of floats

    >>> x = lorenz96([8.0, 8.0, 8.0], 1)
    >>> print(x)
    array([8.0, 8.0, 8.0])

    >>> lorenz96([False, False, True, False, False], 3)
    array([True, False, True, True, True])

    # write your code here to replace return statement
    return NotImplemented
    """
    new_initial = np.array(initial_state)

    while nsteps > 0:
        Xm2 = np.roll(new_initial, 2)
        Xm1 = np.roll(new_initial, 1)
        Xp1 = np.roll(new_initial, -1)
        new_initial = (1/101)*(100 * new_initial + (Xm2-Xp1) * Xm1 + 8)
        nsteps -= 1
    return new_initial


def life(initial_state, nsteps):
    """
    Perform iterations of Conway’s Game of Life.
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of booleans.
    nsteps : int
        Number of steps of Life to perform.
    Returns
    -------
    numpy.ndarray
         Final state of grid in array of booleans
    """

    # write your code here to replace return statement
    init_row, init_col = len(initial_state), len(initial_state[0])
    new_state = np.zeros((init_row, init_col))
    while nsteps > 0:
        for row in range(init_row):
            for col in range(init_col):
                cur_cell = initial_state[row][col]
                sur_cell = number_neighbor(initial_state, init_row,
                                           init_col, row, col)
                print(sur_cell)
                if cur_cell:
                    if sur_cell <= 1 or sur_cell > 3:
                        cur_cell = False
                    elif sur_cell <= 3:
                        cur_cell = True

                else:
                    if sur_cell == 3:
                        cur_cell = True
                    else:
                        cur_cell = False
                new_state[row][col] = cur_cell
        initial_state = np.copy(new_state)
        nsteps -= 1
    return new_state
# function to calculate the number of neighbor which is true


def number_neighbor(x, m, n, row, col):
    count = 0
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if 0 <= i < m and 0 <= j < n and x[i][j]:
                count += 1
    if x[row][col]:
        count -= 1
    return count


def life_periodic(initial_state, nsteps):
    """
    Perform iterations of Conway's Game of Life on a doubly periodic mesh.

    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of booleans.
    nsteps : int
        Number of steps of Life to perform.

    Returns
    -------

    numpy.ndarray
         Final state of grid in array of booleans
    """

# write your code here to replace this return statement
    row, col = len(initial_state), len(initial_state[0])
    while nsteps > 0:
        initial_state = np.array(initial_state)
        new_state = np.zeros((row+2, col+2))
        new_state[1:row+1, 1:col+1] = initial_state
        new_state[1:row+1, -1] = initial_state[:, 0]
        new_state[1:row+1, 0] = initial_state[:, -1]
        new_state[-1, 1:col+1] = initial_state[0, :]
        new_state[0, 1:col+1] = initial_state[-1, :]
        new_state[0, 0] = initial_state[-1, -1]
        new_state[-1, -1] = initial_state[0, 0]
        new_state[0, -1] = initial_state[-1, 0]
        new_state[-1, 0] = initial_state[0, -1]
        new_state = life(new_state, 1)
        initial_state = new_state[1:row+1, 1:col+1]
        nsteps -= 1
    return initial_state


def life2colour(initial_state, nsteps):
    """
    Perform iterations of Conway's Game of Life on a doubly periodic mesh.

    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array ints with value -1, 0, or 1.
        Values of -1 or 1 represent "on" cells of both colours. Zero
        values are "off".
    nsteps : int
        Number of steps of Life to perform.

    Returns
    -------

    numpy.ndarray
        Final state of grid in array of ints of value -1, 0, or 1.
    """

    # write your code here to replace this return statement
    init_row, init_col = len(initial_state), len(initial_state[0])
    initial_state = np.array(initial_state)
    while nsteps > 0:
        new_state = np.zeros((init_row, init_col))
        for row, col in np.ndindex(initial_state.shape):
            count, dom = number_color(initial_state, init_row,
                                      init_col, row, col)
            if initial_state[row][col] != 0:
                if count == 2 or count == 3:
                    new_state[row][col] = initial_state[row][col]
            else:
                if count == 3:
                    new_state[row][col] = dom
        initial_state = new_state
        nsteps -= 1
        print(initial_state)
    return initial_state


def number_color(x, m, n, row, col):
    pos = 0
    neg = 0
    count = 0
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if 0 <= i < m and 0 <= j < n:
                if x[i][j] == 1:
                    pos += 1
                if x[i][j] == -1:
                    neg += 1
    if x[row][col] == 1:
        pos -= 1
    if x[row][col] == -1:
        neg -= 1
    count = pos + neg
    dom = 1 if pos > neg else -1
    return count, dom


def lifepent(initial_state, nsteps):
    """
    Perform iterations of Conway's Game of Life on
    a pentagonal tessellation.

    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial state of grid of pentagons.
    nsteps : int
        Number of steps of Life to perform.

    Returns
    -------

    numpy.ndarray
         Final state of tessellation.
    """

    # write your code here to replace return this statement
    row, col = len(initial_state), len(initial_state[0])
    new_arr = np.zeros((row + 2, col + 4))
    new_arr[1:-1, 2:-2] = initial_state

    while nsteps > 0:
        update_arr = np.zeros((row + 2, col + 4))
        count = 0
        for i in range(1, row + 1):
            for j in range(2, col + 2):
                cur = new_arr[i][j]
                sur = np.sum(new_arr[i - 1:i + 2, j - 1:j + 2])
                count = sur - cur
                count = check_sur(new_arr, count, i, j)
                if cur:
                    if count == 2 or count == 3:
                        update_arr[i][j] = 1
                else:
                    if count == 3 or count == 4 or count == 6:
                        update_arr[i][j] = 1
        new_arr = update_arr
        nsteps -= 1
    return new_arr[1:row + 1, 2:col + 2]


def check_sur(x, count, i, j):
    if i % 2 != 0:
        if j % 2 != 0:
            count -= x[i - 1][j - 1]
        else:
            count -= x[i + 1][j - 1]
    else:
        if j % 2 != 0:
            count -= x[i - 1][j + 1]
        else:
            count -= x[i + 1][j + 1]
    return count

# Remaining routines are for plotting


def plot_lorenz96(data, label=None):
    """
    Plot 1d array on a circle

    Parameters
    ----------
    data: arraylike
        values to be plotted
    label:
        optional label for legend.


    """

    offset = 8

    data = np.asarray(data)
    theta = 2*np.pi*np.arange(len(data))/len(data)

    vector = np.empty((len(data), 2))
    vector[:, 0] = (data+offset)*np.sin(theta)
    vector[:, 1] = (data+offset)*np.cos(theta)

    theta = np.linspace(0, 2*np.pi)

    rings = np.arange(int(np.floor(min(data))-1),
                      int(np.ceil(max(data)))+2)
    for ring in rings:
        plt.plot((ring+offset)*np.cos(theta),
                 (ring+offset)*np.sin(theta), 'k:')

    fig_ax = plt.gca()
    fig_ax.spines['left'].set_position(('data', 0.0))
    fig_ax.spines['bottom'].set_position(('data', 0.0))
    fig_ax.spines['right'].set_color('none')
    fig_ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks(rings+offset, rings)
    plt.fill(vector[:, 0], vector[:, 1],
             label=label, fill=False)
    plt.scatter(vector[:, 0], vector[:, 1], 20)


def plot_array(data, show_axis=False,
               cmap=plt.cm.get_cmap('seismic'), **kwargs):
    """Plot a 1D/2D array in an appropriate format.

    Mostly just a naive wrapper around pcolormesh.

    Parameters
    ----------

    data : array_like
        array to plot
    show_axis: bool, optional
        show axis numbers if true
    cmap : pyplot.colormap or str
        colormap

    Other Parameters
    ----------------

    **kwargs
        Additional arguments passed straight to pyplot.pcolormesh
    """
    plt.pcolormesh(1*data[-1::-1, :], edgecolor='y',
                   vmin=-2, vmax=2, cmap=cmap, **kwargs)

    plt.axis('equal')
    if show_axis:
        plt.axis('on')
    else:
        plt.axis('off')


def plot_pent(x_0, y_0, theta_0, clr=0):
    """
    Plot a pentagram

    Parameters
    ----------
    x_0: float
        x coordinate of centre of the pentegram
    y_0: float
        y coordinate of centre of the pentegram
    theta_0: float
        angle of pentegram (in radians)
    """
    colours = ['w', 'r']
    s_1 = 1/np.sqrt(3)
    s_2 = np.sqrt(1/2)

    theta = np.deg2rad(theta_0)+np.deg2rad([30, 90, 165, 240, 315, 30])
    r_pent = np.array([s_1, s_1, s_2, s_1, s_2, s_1])

    x_pent = x_0+r_pent*np.sin(-theta)
    y_pent = y_0+r_pent*np.cos(-theta)

    plt.fill(x_pent, y_pent, ec='k', fc=colours[clr])


def plot_pents(data):
    """
    Plot pentagrams in Cairo tesselation, coloured by value

    Parameters
    ----------
    data: arraylike
        integer array of values
    """
    plt.axis('off')
    plt.axis('equal')
    data = np.asarray(data).T
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            x_c = (row+1)//2+(row//2)*np.cos(np.pi/6)-(col//2)*np.sin(np.pi/6)
            y_c = (col+1)//2+(col//2)*np.cos(np.pi/6)+(row//2)*np.sin(np.pi/6)
            theta = (90*(row % 2)*((col + 1) % 2)
                     - 90*(row % 2)*(col % 2) - 90*(col % 2))
            clr = data[row, data.shape[1]-1-col]
            plot_pent(x_c, y_c, theta, clr=clr)
