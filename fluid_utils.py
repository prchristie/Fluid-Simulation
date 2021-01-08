"""I understand some of this stuff is going to be hard to read at times. I simply followed a papers
implementation; Real-Time Fluid Dynamics for Games
http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/papers/StamFluidforGames.pdf

This implementation is identical to the one found in that paper, except it is designed for the gpu
to increase performance. The cpu version faced extreme issues on even small grid sizes, whereas
this version's performance is a huge boost on even a reasonable gpu (Im running a 3080, that is not
what I would consider reasonable though).
"""

from typing import Tuple
import numpy as np
from numba import cuda
import math

def diffuse(side_length: int, side: int, current_list: np.ndarray,
            prev_list: np.ndarray, diffusion_rate: float, dt: float):
    """**Diffuses** the values of the current_list outwards by moving the previous list towards the
    values adjacent to each index in the current list

    Args:
        side_length: The length of each side, assuming the lists are square and equal length
        current_list: The current list
        prev_list: The same list but last iteration
        diffusion_rate: How fast the array diffuses
        dt: The timestep taken

    Returns:
        np.ndarray: An updated current_list
    """
    a = dt * diffusion_rate * ((side_length - 2)**2)
    threads_per_block = 32 * 4
    blocks_per_grid = 100
    for i in range(16):
        lin_solve[threads_per_block,
                  blocks_per_grid](side_length, side, current_list, prev_list,
                                   a, 1 + (4 * a))

@cuda.jit
def lin_solve(side_length: int, side: int, current_list: np.ndarray,
              prev_list: np.ndarray, a: float, c: float):
    """Linear solver that works on gpu for Gauss-Seidel relaxation

    Args:
        side_length: A Fluid specific side length, specifying the length of each side of the
            simulated field
        side: The side of the simulated field as an integer
        current_list: The current iteration of the list to be solved
        prev_list: The previous iteration of the list to be solved
        a: No idea what this refers to
        c: No idea what this refers to
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # for _ in range(10):
    for index in range(start, current_list.shape[0], stride):
        x, y = IX_rev(index, side_length)

        if x < 1 or x >= side_length - 1 or y < 1 or y >= side_length - 1:
            continue

        current_list[IX(
            x, y,
            side_length)] = (prev_list[IX(x, y, side_length)] +
                             (a *
                              (current_list[IX(x - 1, y, side_length)] +
                               current_list[IX(x + 1, y, side_length)] +
                               current_list[IX(x, y - 1, side_length)] +
                               current_list[IX(x, y + 1, side_length)]))) / c
    set_bnd(side_length, side, current_list)

@cuda.jit(device=True)
def device_lin_solve(side_length: int, side: int, current_list: np.ndarray,
                     prev_list: np.ndarray, a: float, c: float):
    """The same as above except it runs as a device function so I can use it in kernels.
    No idea if its a good idea, seems to work.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for index in range(start, current_list.shape[0], stride):
        x, y = IX_rev(index, side_length)

        if x < 1 or x >= side_length - 1 or y < 1 or y >= side_length - 1:
            continue

        current_list[IX(
            x, y,
            side_length)] = (prev_list[IX(x, y, side_length)] +
                             (a *
                              (current_list[IX(x - 1, y, side_length)] +
                               current_list[IX(x + 1, y, side_length)] +
                               current_list[IX(x, y - 1, side_length)] +
                               current_list[IX(x, y + 1, side_length)]))) / c
    set_bnd(side_length, side, current_list)

def advect(side_length: int, side: int, current_list: np.ndarray,
           prev_list: np.ndarray, x_velocity: np.ndarray,
           y_velocity: np.ndarray, timestep: float):
    """A nice entry point into the advect function such that the threads per block and blocks per
    grid is hidden

    Args:
        side_length: A Fluid specific side length, specifying the length of each side of the
            simulated field
        side: The side of the simulated field as an integer
        current_list: The current iteration of the list to be advected
        prev_list: The previous iteration of the list to be advected
        x_velocity: Current x velocity
        y_velocity: Current y velocity
        timestep: How long each timestep is. Lower results in a better simulation
    """
    _advect[32 * 8, 100](side_length, side, current_list, prev_list, x_velocity,
                        y_velocity, timestep)

@cuda.jit
def _advect(side_length: int, side: int, current_list: np.ndarray,
            prev_list: np.ndarray, x_velocity: np.ndarray,
            y_velocity: np.ndarray, dt: float):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    dt0 = dt * (side_length - 2)
    for index in range(start, current_list.shape[0], stride):
        i, j = IX_rev(index, side_length)

        if i < 1 or j < 1 or i >= side_length - 1 or j >= side_length - 1:
            continue

        x = i - dt0 * x_velocity[IX(i, j, side_length)]
        y = j - dt0 * y_velocity[IX(i, j, side_length)]

        if x < 0.5:
            x = 0.5
        if x > side_length + 0.5:
            x = side_length + 0.5

        if y < 0.5:
            y = 0.5
        if y > side_length + 0.5:
            y = side_length + 0.5

        i0 = math.floor(x)
        i1 = i0 + 1

        j0 = math.floor(y)
        j1 = j0 + 1

        s1 = x - i0
        s0 = 1.0 - s1

        t1 = y - j0
        t0 = 1 - t1
        current_list[IX(
            i, j, side_length
        )] = s0 * (t0 * prev_list[int(IX(i0, j0, side_length))] +
                   t1 * prev_list[int(IX(i0, j1, side_length))]) + s1 * (
                       t0 * prev_list[int(IX(i1, j0, side_length))] +
                       t1 * prev_list[int(IX(i1, j1, side_length))])
    set_bnd(side_length, side, current_list)


def project(side_length: int, x_velocity: np.ndarray, y_velocity: np.ndarray,
            p: np.ndarray, div: np.ndarray):
    _project[32 * 4, 100](side_length, x_velocity, y_velocity, p, div)


@cuda.jit
def _project(side_length: int, x_velocity: np.ndarray, y_velocity: np.ndarray,
             p: np.ndarray, div: np.ndarray):
    """Used to conserve mass in the velocity field. I dont know what p and div actually refer to. Theoretically will
    perform conservation on any input ndarrays

    Args:
        side_length: The length of each side if the ndarray was 2d
        x_velocity: The current x velocities of the particles
        y_velocity: The current y velocities of the particles
        p (np.ndarray): [description]
        div (np.ndarray): [description]
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    h = 1 / side_length
    for index in range(start, x_velocity.shape[0], stride):
        i, j = IX_rev(index, side_length)

        if i < 1 or j < 1 or i >= side_length - 1 or j >= side_length - 1:
            continue

        div[IX(
            i, j,
            side_length)] = -0.5 * h * (x_velocity[IX(i + 1, j, side_length)] -
                                        x_velocity[IX(i - 1, j, side_length)] +
                                        y_velocity[IX(i, j + 1, side_length)] -
                                        y_velocity[IX(i, j - 1, side_length)])

        p[IX(i, j, side_length)] = 0

    set_bnd(side_length, 0, div)
    set_bnd(side_length, 0, p)

    for i in range(16):
        device_lin_solve(side_length, 0, p, div, 1, 4)

    for index in range(start, x_velocity.shape[0], stride):
        i, j = IX_rev(index, side_length)

        if i < 1 or j < 1 or i >= side_length - 1 or j >= side_length - 1:
            continue

        x_velocity[IX(i, j,
                      side_length)] -= 0.5 * (p[IX(i + 1, j, side_length)] -
                                              p[IX(i - 1, j, side_length)]) / h
        y_velocity[IX(i, j,
                      side_length)] -= 0.5 * (p[IX(i, j + 1, side_length)] -
                                              p[IX(i, j - 1, side_length)]) / h

    set_bnd(side_length, 1, x_velocity)
    set_bnd(side_length, 2, y_velocity)


@cuda.jit(device=True)
def IX(x: int, y: int, N: int) -> int:
    """Converts an x and y position to a 1d index

    Args:
        x: The x pos
        y: y pos
        N: The length of each side of the matrix (IX expects a square matrix)

    Returns:
        int: The 1 dimensional index of of the 2d x,y position.
    """
    return y + (x * N)


@cuda.jit(device=True)
def IX_rev(index: int, N: int) -> Tuple[int, int]:
    """Performs the IX function in reverse
    """
    y = math.floor(index / N)
    x = index % N

    return int(x), int(y)


@cuda.jit(device=True)
def set_bnd(side_length: int, side: int, bounded_array: np.ndarray) -> None:
    """Sets the bounds of an array by setting the borders to a certain value (in this case the
    exact opposite value of the adjacent spot point inwards)

    Args:
        side_length: The length of each side of the square matrix
        side: Which side to set the bound for
        bounded_array: The array to bound
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for index in range(start, bounded_array.shape[0], stride):
        i, j = IX_rev(index, side_length)
        bounded_array[IX(i, 0, side_length)] = -bounded_array[IX(
            i, 1, side_length)] if side == 2 else bounded_array[IX(
                i, 1, side_length)]
        bounded_array[IX(i, side_length - 1, side_length)] = -bounded_array[IX(
            i, side_length - 2, side_length)] if side == 2 else bounded_array[
                IX(i, side_length - 2, side_length)]
        bounded_array[IX(0, j, side_length)] = -bounded_array[IX(
            1, j, side_length)] if side == 1 else bounded_array[IX(
                1, j, side_length)]
        bounded_array[IX(side_length - 1, j, side_length)] = -bounded_array[IX(
            side_length - 2, j, side_length)] if side == 1 else bounded_array[
                IX(side_length - 2, j, side_length)]

    # 4 corners
    bounded_array[IX(
        0, 0, side_length)] = 0.5 * (bounded_array[IX(1, 0, side_length)] +
                                     bounded_array[IX(0, 1, side_length)])
    bounded_array[IX(0, side_length - 1, side_length)] = 0.5 * (
        bounded_array[IX(1, side_length - 1, side_length)] +
        bounded_array[IX(0, side_length - 2, side_length)])
    bounded_array[IX(side_length - 1, 0, side_length)] = 0.5 * (
        bounded_array[IX(side_length - 2, 0, side_length)] +
        bounded_array[IX(side_length - 1, 1, side_length)])
    bounded_array[IX(side_length - 1, side_length - 1, side_length)] = 0.5 * (
        bounded_array[IX(side_length - 2, side_length - 1, side_length)] +
        bounded_array[IX(side_length - 1, side_length - 2, side_length)])


def IX_rev_cpu(index: int, N: int) -> Tuple[int, int]:
    """IX_rev but on the cpu
    """
    y = math.floor(index / N)
    x = index % N

    return int(x), int(y)


def IX_cpu(x: int, y: int, N: int) -> int:
    """IX but on the cpu
    """
    return y + (x * N)


def diffuse_cpu(side_length: int, side: int, current_list: np.ndarray,
                prev_list: np.ndarray, diff: float, dt: float):
    """Testing implementation of diffuse to compare results, so that I could think about code in
    the traditional way instead of the insanely multithreaded way.
    """
    a = dt * diff * (side_length - 2) * (side_length - 2)
    for n in range(5):
        for i in range(1, side_length - 1):
            for j in range(1, side_length - 1):
                current_list[IX_cpu(i, j, side_length)] = (
                    prev_list[IX_cpu(i, j, side_length)] + a *
                    (current_list[IX_cpu(i - 1, j, side_length)] +
                     current_list[IX_cpu(i + 1, j, side_length)] +
                     current_list[IX_cpu(i, j - 1, side_length)] +
                     current_list[IX_cpu(i, j + 1, side_length)])) / (1 +
                                                                      4 * a)

    return current_list.copy(), prev_list
