import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.widgets import Slider
import copy
import pickle
import math
import time
from numba import jit

matplotlib.use('QtAgg')

G = 6.674e-11  # m3 kg-1 s-2
AU = 1.496e+11 #m

MASS_SUN = 1.979e30  # kg
G_TIMES_MASS_SUN = G * MASS_SUN  # m^3 s^-2

MASS_JUPITER = 1.898e27  # kg
DISTANCE_JUPITER = 747.89e9  # m
ECCENTRICITY_JUPITER = 0.049

MASS_SATURN = 5.683e26  # kg
DISTANCE_SATURN = 1.4507e12  # m

MASS_NEPTUNE = 1.024e26  # kg

MASS_URANUS = 8.681e25  # kg

M_EARTH = 5.972e24  #k kg
DISTANCE_NEPTUNE = 4.4925e9  # km
SIGMA = M_EARTH * 10**-5 / (np.pi * DISTANCE_NEPTUNE ** 2)

SECONDS_IN_YEAR = 3.15e7


def numba_run(position_array, velocity_array, acceleration_array, gravity_interaction_constant, infinite_diagonal, number_of_iterations=10**6, dt=1e6):
    n_bodies = np.shape(position_array)[0]

    position_saves = np.zeros((number_of_iterations + 1, n_bodies, 3))
    velocity_saves = np.zeros((number_of_iterations + 1, n_bodies, 3))
    time_saves = [dt * i for i in range(number_of_iterations + 1)]

    position_saves[0] = position_array
    velocity_saves[0] = velocity_array
    for i in range(number_of_iterations):
        position_array += velocity_array * dt + .5 * acceleration_array * dt ** 2

        delta_r_matrix = (position_array[np.newaxis, :] - position_array[:, np.newaxis])
        delta_r_matrix_to_the_minus_3 = (delta_r_matrix[:,:,0]**2 + delta_r_matrix[:,:,1]**2 + delta_r_matrix[:,:,2]**2 + infinite_diagonal) ** (-1.5)
        accelerations = np.sum((gravity_interaction_constant * delta_r_matrix_to_the_minus_3)[:, :, np.newaxis]
                               * delta_r_matrix, 1)

        velocity_array += .5 * (accelerations + acceleration_array) * dt
        acceleration_array = accelerations

        position_saves[i+1] = position_array
        velocity_saves[i+1] = velocity_array

    return position_saves, velocity_saves, time_saves


def get_acceleration(position_array, gravity_interaction_constant, infinite_diagonal):
    delta_r_matrix = position_array[np.newaxis, :] - position_array[:, np.newaxis]
    delta_r_matrix_to_the_minus_3 = (delta_r_matrix[:, :, 0] ** 2 +
                                     delta_r_matrix[:, :, 1] ** 2 + infinite_diagonal) ** (-1.5)
    return np.sum((gravity_interaction_constant * delta_r_matrix_to_the_minus_3)[:, :, np.newaxis]
                                   * delta_r_matrix, 1)


def rotation_matrix(angle, axis='z'):
    if axis == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    if axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    if axis == 'x':
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def orbit(a, e=0, Omega=None, omega=None, i=None):
    if e is None:
        e = np.random.uniform(0, 0.1)
    if omega is None:
        omega = np.random.uniform(0, 2*np.pi)
    if i is None:
        i = np.random.uniform(-0.3, 0.3)
    if Omega is None:
        Omega = np.random.uniform(0, 2*np.pi)
    rotated_system_matrix = rotation_matrix(omega)
    return (np.matmul(rotated_system_matrix, np.array([a*(1-e), 0, 0])),
            np.matmul(rotated_system_matrix, np.array([0, np.sqrt(G_TIMES_MASS_SUN * (1+e) / (a * (1-e))), 0])))


def nice_model_objects():
    uranus_radius = np.random.uniform(11, 13)
    neptune_radius = np.random.uniform(max(uranus_radius + 2, 13.5), 17)
    radius_list = [5.45 * AU, 8.65 * AU, uranus_radius * AU, neptune_radius * AU]

    position_list = [0] * 4
    velocity_list = [0] * 4
    for i, radius in enumerate(radius_list):
        position_list[i] = orbit(radius)[0]
        velocity_list[i] = orbit(radius)[1]

    mass_list = [MASS_JUPITER, MASS_SATURN, MASS_URANUS, MASS_NEPTUNE]

    return position_list, velocity_list, mass_list


def initialise_system(position_list, velocity_list, mass_list):
    n_bodies = len(mass_list) + 1

    position_array = np.zeros([n_bodies, 3])
    velocity_array = np.zeros([n_bodies, 3])
    mass_vector = np.zeros(n_bodies)

    position_sun = np.zeros(3)
    for i in range(n_bodies - 1):
        position_sun -= mass_list[i] * position_list[i] / MASS_SUN

    velocity_sun = np.zeros(3)
    for i in range(n_bodies - 1):
        velocity_sun -= mass_list[i] * velocity_list[i] / MASS_SUN

    position_array[0] = position_sun
    velocity_array[0] = velocity_sun
    mass_vector[0] = MASS_SUN

    position_array[1:] = np.array(position_list)
    velocity_array[1:] = np.array(velocity_list)
    mass_vector[1:] = np.array(mass_list)

    n_bodies = len(mass_vector)

    infinite_diagonal = np.zeros([n_bodies, n_bodies])
    for i in range(n_bodies):
        infinite_diagonal[i, i] = math.inf
    gravity_interaction_constant = G * mass_vector[np.newaxis, :]

    delta_r_matrix = (position_array[np.newaxis, :] - position_array[:, np.newaxis])
    delta_r_matrix_to_the_minus_3 = (delta_r_matrix[:, :, 0]**2 +
                                     delta_r_matrix[:, :, 1]**2 + delta_r_matrix[:, :, 2]**2 + infinite_diagonal) ** (-1.5)
    acceleration_array = np.sum((gravity_interaction_constant * delta_r_matrix_to_the_minus_3)[:, :, np.newaxis]
                                * delta_r_matrix, 1)

    print(acceleration_array)
    return position_array, velocity_array, acceleration_array, gravity_interaction_constant, infinite_diagonal


def run_gravity_simulation(position_list, velocity_list, mass_list, number_of_iterations, dt):
    position_array, velocity_array, acceleration_array, gravity_interaction_constant, infinite_diagonal = initialise_system(
        position_list, velocity_list, mass_list)
    return numba_run(position_array, velocity_array, acceleration_array, gravity_interaction_constant, infinite_diagonal, number_of_iterations, dt)


def main():
    dt = 1e5
    total_time = 1e2  # years
    number_of_iterations = 1000
    # int(total_time * SECONDS_IN_YEAR / dt)

    position_list, velocity_list, mass_list = nice_model_objects()
    position_saves, velocity_saves, time_saves = run_gravity_simulation(position_list, velocity_list, mass_list, number_of_iterations, dt)

    name = 'Test'
    np.save('saves/' + name + '_position', position_saves, allow_pickle=True)
    np.save('saves/' + name + '_velocity', velocity_saves, allow_pickle=True)
    np.save('saves/' + name + '_time', time_saves, allow_pickle=True)


if __name__ == "__main__":
    main()
