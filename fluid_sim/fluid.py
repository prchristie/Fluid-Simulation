import numpy as np
from numba import cuda
from fluid_sim.fluid_utils import IX_cpu, diffuse, advect, project


class Fluid:
    def __init__(self, grid_size: int, dt: float, diffusion_rate: float,
                 viscosity: float):
        """Initializes a fluid board, where the grid size is the square root of the total number of
        simulatenously simulated grid squares of fluid.

        This fluid class stores privately simulated matrices in cuda devices to increase
        performance. The three user effected matrices (density, x and y velocity) are not kept as
        devices because they are indexed and altered very often and from my testing it is more
        performant to simply copy the entire list over every step than indexing and updating a
        device directly.

        To get the density and velocities as they update, use the density and velocities as a
        public variable (Fluid.density/x_vel/y_vel). They will always have the most up to date
        instance of the values.

        Args:
            grid_size: The length of each side of the simulated board.
            dt: How long each timestep is.
            diffusion_rate: The rate at which the fluid dissolves into lesser occupied space.
            viscosity: How thick the fluid is.
        """
        self.N = grid_size
        self.grid_space = (grid_size)**2
        self.dt = dt
        self.diffusion_rate = diffusion_rate
        self.viscosity = viscosity

        self.density = np.zeros(self.grid_space)
        self._prev_density = cuda.to_device(np.zeros(self.grid_space))

        self.vel_x = np.zeros(self.grid_space)
        self.vel_y = np.zeros(self.grid_space)

        self._prev_vel_x = cuda.to_device(np.zeros(self.grid_space))
        self._prev_vel_y = cuda.to_device(np.zeros(self.grid_space))

    def add_density(self, x: int, y: int, amount: float) -> None:
        """Adds density to the (x, y) position on the fluid board.

        Args:
            x: The x position.
            y: The y position.
            amount: The amount of fluid that will be added.
        """
        index = IX_cpu(x, y, self.N)
        self.density[index] += amount

    def add_velocity(self, x: int, y: int, amount_x: float,
                     amount_y: float) -> None:
        """Adds velocity to the (x, y) position on the fluid board. Velocity moves density around.

        Args:
            x: The x position.
            y: The y position.
            amount_x: The amount of velocity in the x direction.
            amount_y: The amount of velocity in the y direction.
        """
        index = IX_cpu(x, y, self.N)
        self.vel_x[index] += amount_x
        self.vel_y[index] += amount_y

    def _density_step(self, density, vel_x, vel_y) -> None:
        diffuse(self.N, 0, self._prev_density, density, self.diffusion_rate,
                self.dt)
        advect(self.N, 0, density, self._prev_density, vel_x, vel_y, self.dt)

    def density_step(self) -> None:
        """An 'optimised' density step that only copies the things to device that are truly needed
        for a density step. This should NOT be used if doing density and velocity steps together
        (which you should be if you are trying to simulate correctly). User step() instead.
        """
        density_device = cuda.to_device(self.density)
        vel_x_device = cuda.to_device(self.vel_x)
        vel_y_device = cuda.to_device(self.vel_y)
        self._density_step(density_device, vel_x_device, vel_y_device)
        cuda.synchronize()
        self.density = density_device.copy_to_host()

    def vel_step(self) -> None:
        """An 'optimised' velocity step that only copies the things to device that are truly needed
        for a velocity step. This should NOT be used if doing density and velocity steps together
        (which you should be if you are trying to simulate correctly). Use step() instead.
        """
        vel_x_device = cuda.to_device(self.vel_x)
        vel_y_device = cuda.to_device(self.vel_y)
        self._vel_step(vel_x_device, vel_y_device)

        self.vel_x = vel_x_device.copy_to_host()
        self.vel_y = vel_y_device.copy_to_host()

    def _vel_step(self, vel_x, vel_y) -> None:
        diffuse(self.N, 1, self._prev_vel_x, vel_x, self.viscosity, self.dt)
        diffuse(self.N, 2, self._prev_vel_y, vel_y, self.viscosity, self.dt)

        project(self.N, self._prev_vel_x, self._prev_vel_y, vel_x, vel_y)

        advect(self.N, 1, vel_x, self._prev_vel_x, self._prev_vel_x,
               self._prev_vel_y, self.dt)
        advect(self.N, 2, vel_y, self._prev_vel_y, self._prev_vel_x,
               self._prev_vel_y, self.dt)

        project(self.N, vel_x, vel_y, self._prev_vel_x, self._prev_vel_y)

    def step(self) -> None:
        """A total step of the simulation. First simulates the velocity of the fluid environment,
        then the effect of that velocity on the density.

        I have found that by far the biggest hit to my performance is the fact I need to copy the
        density and velocities to and from the gpu every step if I plan to display the fluid each
        frame.

        If there is a better way, please please tell me. While I still get 25x more frames on gpu
        over cpu, it'd be cool to reduce this problem as without it I could hit much higher fps.
        """
        vel_x_device = cuda.to_device(self.vel_x)
        vel_y_device = cuda.to_device(self.vel_y)
        density_device = cuda.to_device(self.density)

        self._vel_step(vel_x_device, vel_y_device)
        self._density_step(density_device, vel_x_device, vel_y_device)

        self.density = density_device.copy_to_host()
        self.vel_x = vel_x_device.copy_to_host()
        self.vel_y = vel_y_device.copy_to_host()
