"""A very rushed and simple interface that shows what the fluid sim does
"""

from fluid_sim.fluid import Fluid
import os
from numba import cuda
import random

from pygame import surfarray
import pygame

# So that pygame dopesnt initialize sound (only useful on wsl systems i think)
os.environ['SDL_AUDIODRIVER'] = 'dsp'


def main():
    DIMENSION = 600

    WIDTH = HEIGHT = DIMENSION

    pygame.init()
    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    fluid = Fluid(WIDTH, 0.05, 0.0000001, 0.0000001)

    prev_x, prev_y = pygame.mouse.get_pos()
    running = True
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEMOTION:
                x, y = pygame.mouse.get_pos()
                change_x = x - prev_x
                change_y = y - prev_y
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        fluid.add_density(x + i, y + j,
                                          random.randint(100, 200))
                        fluid.add_velocity(x, y, change_x * 2, change_y * 2)

                prev_x, prev_y = x, y

        fluid.density[fluid.density > 0] -= 0.5

        fluid.step()
        density = fluid.density.reshape(WIDTH, HEIGHT).copy()
        surfarray.blit_array(screen, density)

        fps = font.render(str(int(clock.get_fps())), True,
                          pygame.Color('White'))
        screen.blit(fps, (50, 50))

        # Flip the display
        pygame.display.flip()
        clock.tick(144)

    pygame.quit()
