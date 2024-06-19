import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import copy
import pickle
import math
import time
import itertools


matplotlib.use('QtAgg')

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


class GravityModel:
    def __init__(self, mass_bodies):
        if isinstance(mass_bodies, list):
            self.mass_bodies = mass_bodies
        elif isinstance(mass_bodies, PointMass):
            self.mass_bodies = [mass_bodies]
        else:
            raise Exception("Bodies invalid type")

        self.t = 0
        self.saves = []

        #calculate Sun
        r_cm = np.array([0., 0.])
        v_cm = np.array([0., 0.])
        for mass_body in self.mass_bodies:
            r_cm += mass_body.mass * mass_body.position
            v_cm += mass_body.mass * mass_body.velocity

        self.mass_bodies.insert(0, PointMass(-r_cm / MASS_SUN, -v_cm / MASS_SUN, MASS_SUN))

        #initialise position and velocity vectors
        self.n_bodies = len(self.mass_bodies)
        self.mass_body_positions = np.zeros([self.n_bodies, 2])
        self.mass_body_velocities = np.zeros([self.n_bodies, 2])
        for i, body in enumerate(mass_bodies):
            self.mass_body_positions[i] = body.position
            self.mass_body_velocities[i] = body.velocity

        #initialise G m_i m_j matrix for gravity calculations
        self.mass_vector = np.array([mass_body.mass for mass_body in self.mass_bodies])
        self.gravity_interaction_constant = G * np.tensordot(np.ones(self.n_bodies), self.mass_vector, axes=0)

        #initialise accelaration for leapfrog
        infinite_diagonal = np.zeros((self.n_bodies, self.n_bodies))
        np.fill_diagonal(infinite_diagonal, math.inf)
        self.infinite_diagonal = infinite_diagonal #need this later with gravity calculations
        self.ones = np.ones(self.n_bodies)
        self.mass_body_accelerations = self.get_acceleration()

    def get_acceleration(self):
        delta_r_matrix = self.mass_body_positions[np.newaxis, :] - self.mass_body_positions[:, np.newaxis]
        delta_r_matrix_to_the_minus_3 = (delta_r_matrix[:, :, 0] ** 2 +
                                         delta_r_matrix[:, :, 1] ** 2 + self.infinite_diagonal) ** (-1.5)
        return np.einsum('i,jik', self.ones,
                         (self.gravity_interaction_constant * delta_r_matrix_to_the_minus_3)[:, :, np.newaxis]
                         * delta_r_matrix)

    def time_update(self, dt):
        self.mass_body_positions += self.mass_body_velocities * dt + .5 * self.mass_body_accelerations * dt ** 2

        # leapfrog update x and v
        accelerations = self.get_acceleration()

        self.mass_body_velocities += .5 * (accelerations + self.mass_body_accelerations) * dt
        self.mass_body_accelerations = accelerations

        self.t += dt

    def save(self):
        bodies_for_save = [0] * self.n_bodies
        for i, body in enumerate(self.mass_bodies):
            body.position = self.mass_body_positions[i]
            body.velocity = self.mass_body_velocities[i]
            bodies_for_save[i] = copy.deepcopy(body)
        self.saves += [(self.t / SECONDS_IN_YEAR, bodies_for_save)]

    def run(self, dt, number_of_iterations, number_of_saves=0):
        print("Starting simulation with " + str(self.n_bodies) + ' bodies, dt=' + str(dt) + "s and "
              + str(number_of_iterations) + " iterations")
        start = time.time()

        if number_of_saves == 0:
            save_list = [False] * number_of_iterations
        elif number_of_saves > 0:
            iterations_per_save = number_of_iterations // number_of_saves
            save_list = ([True] + [False]*(iterations_per_save-1))*number_of_saves + [True]

        for save_value in save_list:
            if save_value:
                self.save()

            self.time_update(dt)

        end = time.time()
        return end-start


class ModelSaves:
    def __init__(self, saves):
        self.saves = saves
        self.time_value_saves = [snapshot[0] for snapshot in self.saves]
        self.point_mass_objects_saves = [snapshot[1] for snapshot in self.saves]

    def store_saves(self, filename):
        with open(filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def show_states(self):
        if len(self.point_mass_objects_saves) == 0:
            raise ValueError("Trying to plot no saves")

        fig = plt.figure(figsize=[8, 8])

        fax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        sax = fig.add_axes([0.2, 0.9, 0.7, 0.1])

        s_t = Slider(ax=sax, label='data number', valmin=0, valmax=len(self.saves) - 1, valinit=0)

        maxval = 30 * AU

        def update(val):
            fax.cla()

            for mass_object in self.point_mass_objects_saves[int(s_t.val)]:
                a = mass_object.calculate_semi_major()
                e = mass_object.calculate_eccentricity()
                varpi = mass_object.calculate_varpi()
                b = a * np.sqrt(1 - e*e)

                kakudo = np.linspace(0, 2*np.pi, 100)

                x_ellipse = [np.cos(varpi) * (a * np.cos(angle) - a * e) - np.sin(varpi) * b * np.sin(angle) for angle in kakudo]
                y_ellipse = [np.sin(varpi) * (a * np.cos(angle) - a * e) + np.cos(varpi) * b * np.sin(angle) for angle in kakudo]

                fax.plot(x_ellipse, y_ellipse)
                fax.scatter(mass_object.position[0], mass_object.position[1])

            k = 1.2  # thickness border
            fax.set_xlim(-k * maxval, k * maxval)
            fax.set_ylim(-k * maxval, k * maxval)
            fax.set_aspect('equal', adjustable='box')

        s_t.on_changed(update)
        update(0)
        plt.show()

    def plot_path(self, targets=[0,1]):
        true_target = targets[0]
        reference_target = targets[1]

        angles = [snapshot[reference_target].calculate_pi() for snapshot in self.point_mass_objects_saves]

        rotated_vectors = [np.matmul(rotation_matrix(-angle), self.point_mass_objects_saves[i][true_target].position) for i, angle in enumerate(angles)]
        x_values = [vector[0] for vector in rotated_vectors]
        y_values = [vector[1] for vector in rotated_vectors]

        minx = np.min(x_values)
        maxx = np.max(x_values)
        miny = np.min(y_values)
        maxy = np.max(y_values)
        maxval = max(abs(maxx), abs(maxy), abs(minx), abs(miny))

        plt.scatter(x_values, y_values)
        ax = plt.gca()
        k = 1.2  # thickness border
        ax.set_xlim(-k * maxval, k * maxval)
        ax.set_ylim(-k * maxval, k * maxval)
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel("distance (m)")
        plt.ylabel("distance (m)")
        plt.show()

    def plot_varpis(self, targets=None):
        if targets is None:
            targets = range(1, len(self.point_mass_objects_saves[0]))

        for i in targets:
            varpis = [snapshot[i].calculate_varpi() for snapshot in self.point_mass_objects_saves]
            plt.plot(self.time_value_saves, varpis, marker='o', label='planet ' + str(i))

        axes = plt.gca()
        axes.set_ybound(lower=-4, upper=4)
        plt.legend()
        plt.xlabel("time (yr)")
        plt.ylabel("angle of periapsis (rad)")
        plt.show()

    def plot_resonance_ratio(self, targets):
        target_1 = targets[0]
        target_2 = targets[1]

        semi_major_1 = np.array([snapshot[target_1].calculate_semi_major() for snapshot in self.point_mass_objects_saves])
        semi_major_2 = np.array([snapshot[target_2].calculate_semi_major() for snapshot in self.point_mass_objects_saves])

        plt.plot((semi_major_2 / semi_major_1) ** (3 / 2))
        axes = plt.gca()
        axes.set_ybound(lower=0)
        plt.xlabel("time (yr)")
        plt.ylabel("$\mathregular{P_2}/\mathregular{P_1}$")
        plt.show()

    def plot_semi_majors(self, targets=None):
        if targets is None:
            targets = range(1, len(self.point_mass_objects_saves[0]))

        max_semi_major = 0
        for i in targets:
            semi_majors = [snapshot[i].calculate_semi_major()/AU for snapshot in self.point_mass_objects_saves]
            max_semi_major_i = max(semi_majors)
            if max_semi_major_i > max_semi_major:
                max_semi_major = max_semi_major_i
            plt.plot(self.time_value_saves, semi_majors, marker='o', label='planet ' + str(i))

        axes = plt.gca()
        axes.set_ybound(lower=0, upper=1.3 * max_semi_major)
        plt.legend()
        plt.xlabel("time (yr)")
        plt.ylabel("semi-major (AU)")
        plt.show()

    def plot_distances(self, targets=None):
        if targets is None:
            targets = range(1, len(self.point_mass_objects_saves[0]))

        plt.figure()
        plt.clf()
        ax=plt.gca()

        for i in targets:
            semi_majors = [snapshot[i].calculate_semi_major()/AU for snapshot in self.point_mass_objects_saves]
            min_distance = [snapshot[i].calculate_semi_major() * (1 - snapshot[i].calculate_eccentricity())/AU for
                            snapshot in self.point_mass_objects_saves]
            max_distance = [snapshot[i].calculate_semi_major() * (1 + snapshot[i].calculate_eccentricity()) / AU for
                            snapshot in self.point_mass_objects_saves]

            color = next(ax._get_lines.prop_cycler)['color']
            for list_to_plot in [semi_majors, min_distance, max_distance]:
                if list_to_plot == semi_majors:
                    plt.plot(self.time_value_saves, list_to_plot, marker='o', label='planet ' + str(i), color=color,
                             markersize=.5)
                else:
                    plt.plot(self.time_value_saves, list_to_plot, marker='o', color=color,
                             markersize=.5)

        ax = plt.gca()
        ax.set_ybound(lower=0, upper=30)
        plt.legend()
        plt.xlabel("time (yr)")
        plt.ylabel("$a/q/Q$ (AU)")
        plt.show()

    def plot_eccentricities(self, targets=None):
        if targets is None:
            targets = range(1, len(self.point_mass_objects_saves[0]))

        max_eccentricity = 0
        for i in targets:
            eccentricities = [snapshot[i].calculate_eccentricity() for snapshot in self.point_mass_objects_saves]
            max_eccentricity_i = max(eccentricities)
            if max_eccentricity_i > max_eccentricity:
                max_eccentricity = max_eccentricity_i
            plt.plot(self.time_value_saves, eccentricities, marker='o', label='planet ' + str(i))

        axes = plt.gca()
        axes.set_ybound(lower=-0.1, upper=1.1)
        plt.xlabel("time (yr)")
        plt.ylabel("eccentricity")
        plt.legend()
        plt.show()

    def plot_total_energy(self):
        number_of_saves = len(self.saves)
        energy_without_sun = np.zeros(number_of_saves)
        energy_with_sun = np.zeros(number_of_saves)
        for i, snapshot in enumerate(self.point_mass_objects_saves):
            energy = 0
            for body in snapshot[1:]:
                energy += 1/2 * body.mass * np.dot(body.velocity, body.velocity)

            for body1, body2 in itertools.combinations(snapshot, 2):
                r = np.linalg.norm(body1.position - body2.position)
                energy -= G * body1.mass * body2.mass / r

            energy_without_sun[i] = energy
            energy_with_sun[i] = energy + 1/2 * snapshot[0].mass * np.dot(snapshot[0].velocity, snapshot[0].velocity)

        plt.plot(self.time_value_saves, energy_without_sun, marker='o', label="energy without Sun")
        plt.plot(self.time_value_saves, energy_with_sun, marker='o', label="energy with Sun")

        axes = plt.gca()
        axes.set_ybound(upper = 0)
        plt.xlabel("time (yr)")
        plt.ylabel("energy (J)")
        plt.legend()
        plt.show()



class PointMass:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass

    @classmethod
    def orbit(cls, distance, mass, angle=0, eccentricity=0):
        velocity = np.sqrt(G_TIMES_MASS_SUN * (1 + eccentricity) / distance)
        return cls(np.array([np.cos(angle) * distance, np.sin(angle) * distance]),
                   np.array([-np.sin(angle) * velocity, np.cos(angle) * velocity]), mass)

    def __str__(self):
        return str(
            "position = " + str(self.position) + ", velocity = " + str(self.velocity) + ",mass = " + str(self.mass))

    def gravity(self, other_object):
        r = other_object.position - self.position
        abs_r = np.linalg.norm(r)
        return G * self.mass * other_object.mass * r / (abs_r * abs_r * abs_r)

    def calculate_energy_density(self):
        position_norm = np.linalg.norm(self.position)
        velocity_norm = np.linalg.norm(self.velocity)
        return - G_TIMES_MASS_SUN / position_norm + velocity_norm * velocity_norm / 2

    def calculate_angular_momentum_density(self):
        return np.cross(self.position, self.velocity)

    def eccentricity_vector(self):
        h = self.calculate_angular_momentum_density()
        abs_r = np.linalg.norm(self.position)
        return h * np.matmul(rotation_matrix(-np.pi / 2), self.velocity) \
               / (G_TIMES_MASS_SUN) - self.position / abs_r

    def calculate_semi_major(self):
        return -G_TIMES_MASS_SUN / (2 * self.calculate_energy_density())

    def calculate_eccentricity(self):
        return np.linalg.norm(self.eccentricity_vector())

    def calculate_varpi(self):
        eccentricity_vector = self.eccentricity_vector()
        return math.atan2(eccentricity_vector[1], eccentricity_vector[0])


def build_resonance_chain(resonances, eccentricities=None, angles=None, distance=DISTANCE_JUPITER, masses=None, random=False):
    N = len(resonances)

    if random:
        if eccentricities == None:
            eccentricities = np.random.uniform(0., .1, size=(N + 1))
        if angles == None:
            angles = np.random.uniform(0., 2 * np.pi, size=(N + 1))
        if masses == None:
            masses = np.ones(N + 1) * MASS_JUPITER
    else:
        if eccentricities == None:
            eccentricities = np.zeros(N + 1)
        if angles == None:
            angles = np.zeros(N + 1)
        if masses == None:
            masses = np.ones(N + 1) * MASS_JUPITER

    if not (N + 1 == len(eccentricities) and N + 1 == len(angles)) and N + 1 == len(masses):
        raise Exception("Lengths don't match up")

    objects = []
    objects += [PointMass.orbit(distance, masses[0], eccentricity=eccentricities[0], angle=angles[0])]
    for i in range(N):
        distance = (1 - eccentricities[i + 1]) * (distance / (1 - eccentricities[i])) * resonances[i] ** (2 / 3)
        objects += [PointMass.orbit(distance, masses[i + 1], eccentricity=eccentricities[i + 1], angle=angles[i + 1])]

    return objects


def rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def numba_loop(model, dt=1e6, number_of_iterations=10**6, number_of_saves=0):
    if number_of_saves == 0:
        model.mass_body_positions, model.mass_body_velocities, model.mass_body_accelerations = (
            run_numba(model.mass_body_positions, model.mass_body_velocities, model.mass_body_accelerations,
                  model.mass_vector, model.gravity_interaction_constant, model.infinite_diagonal,
                  dt, number_of_iterations))
        model.t += dt * number_of_iterations

    if number_of_saves > 0:
        iterations_per_save = number_of_iterations // number_of_saves

        model.save()
        for i in range(number_of_saves):
            model.mass_body_positions, model.mass_body_velocities, model.mass_body_accelerations = (
                run_numba(model.mass_body_positions, model.mass_body_velocities,model.mass_body_accelerations,
                          model.mass_vector, model.gravity_interaction_constant, model.infinite_diagonal,
                          dt, iterations_per_save))
            model.t += dt*iterations_per_save
            model.save()
            print("Saved " + str(i + 1) + "/" + str(number_of_saves))

@jit
def run_numba(position_array, velocity_array, acceleration_array, mass_array, gravity_interaction_constant,
              infinite_diagonal, dt=1e6, number_of_iterations=10**6):
    FRICTION_CONSTANT = (G * SIGMA * (MASS_SUN / mass_array) ** (2/3))[:, np.newaxis]
    for i in range(number_of_iterations):
        position_array += velocity_array * dt + .5 * acceleration_array * dt ** 2

        # leapfrog update x and v
        delta_r_matrix = position_array[np.newaxis, :] - position_array[:, np.newaxis]
        delta_r_matrix_to_the_minus_3 = (delta_r_matrix[:, :, 0] ** 2 +
                                         delta_r_matrix[:, :, 1] ** 2 + infinite_diagonal) ** (-1.5)
        gravity_accelerations = np.sum((gravity_interaction_constant * delta_r_matrix_to_the_minus_3)[:, :, np.newaxis]
                         * delta_r_matrix, 1)
        friction_accelerations = FRICTION_CONSTANT * velocity_array / (np.sqrt(np.sum(velocity_array**2, axis=1))[:, np.newaxis])
        accelerations = gravity_accelerations + friction_accelerations

        velocity_array += .5 * (accelerations + acceleration_array) * dt
        acceleration_array = accelerations
    return position_array, velocity_array, acceleration_array


def nice_model_objects():
    jupiter = PointMass.orbit(5.45 * AU, MASS_JUPITER)
    saturn = PointMass.orbit(8.65 * AU, MASS_SATURN)
    neptune_radius = np.random.uniform(11, 13)
    uranus = PointMass.orbit(neptune_radius * AU, MASS_URANUS)
    uranus_radius = np.random.uniform(max(neptune_radius + 2, 13.5), 17)
    neptune = PointMass.orbit(uranus_radius * AU, MASS_NEPTUNE)
    return [jupiter, saturn, uranus, neptune]


def main():
    dt = 1e6
    total_time = 1e5  # years
    number_of_saves = 10 ** 3

    number_of_iterations = int(total_time * SECONDS_IN_YEAR / dt)

    objects = nice_model_objects()
    model = GravityModel(objects)

    numba_loop(model, dt, number_of_iterations, number_of_saves)

    saves = ModelSaves(model.saves)
    saves.store_saves("Test")


if __name__ == "__main__":
    main()
