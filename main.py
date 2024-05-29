import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.widgets import Slider
import copy
import pickle

matplotlib.use('QtAgg')

G = 6.674e-11  # m3 kg-1 s-2
MASS_SUN = 1.979e30  # kg
G_TIMES_MASS_SUN = G * MASS_SUN  # m^3 s^-2

MASS_JUPITER = 1.898e27  # kg
DISTANCE_JUPITER = 747.89e9  # m
ECCENTRICITY_JUPITER = 0.049

MASS_SATURN = 5.683e26  # kg
DISTANCE_SATURN = 1.4507e12  # m

MASS_NEPTUNE = 1.024e26  # kg

MASS_URANUS = 8.681e25  # kg

SECONDS_IN_YEAR = 3.15e7


class GravityModel:
    def __init__(self, objects):
        if isinstance(objects, list):
            self.point_mass_objects = objects
        elif isinstance(objects, PointMass):
            self.point_mass_objects = [objects]
        else:
            raise Exception("Objects invalid type")

        self.t = 0
        self.initialised = False

    def __str__(self):
        return '\n'.join([str(i) for i in self.point_mass_objects])

    def position_state(self):
        plt.figure()
        M = len(self.point_mass_objects)
        scatters = np.zeros((M, 2))
        for i, object in enumerate(self.point_mass_objects):
            scatters[i] = object.position
        plt.scatter(scatters[:, 0], scatters[:, 1])
        plt.show()

    def initialise(self):
        r_cm = np.array([0., 0.])
        v_cm = np.array([0., 0.])
        for object in self.point_mass_objects:
            r_cm += object.mass * object.position
            v_cm += object.mass * object.velocity

        self.point_mass_objects.insert(0, PointMass(-r_cm / MASS_SUN, -v_cm / MASS_SUN, MASS_SUN))

        forces = np.zeros((len(self.point_mass_objects), 2))
        for (i, j) in combinations(range(len(self.point_mass_objects)), 2):
            object_main = self.point_mass_objects[i]
            object_other = self.point_mass_objects[j]
            force = object_main.gravity(object_other)
            forces[i] += force
            forces[j] -= force

        for i in range(len(self.point_mass_objects)):
            mass_object = self.point_mass_objects[i]
            mass_object.acceleration = forces[i] / mass_object.mass

    def time_update(self, dt):
        for mass_object in self.point_mass_objects:
            mass_object.position += mass_object.velocity * dt + 1 / 2 * mass_object.acceleration * dt ** 2

        # calculate forces brute force
        forces = np.zeros((len(self.point_mass_objects), 2))
        for (i, j) in combinations(range(len(self.point_mass_objects)), 2):
            object_main = self.point_mass_objects[i]
            object_other = self.point_mass_objects[j]
            force = object_main.gravity(object_other)
            forces[i] += force
            forces[j] -= force

        # leapfrog update x and v
        for i in range(len(self.point_mass_objects)):
            mass_object = self.point_mass_objects[i]
            acceleration = forces[i] / mass_object.mass
            mass_object.velocity += 1 / 2 * (mass_object.acceleration + acceleration) * dt
            mass_object.acceleration = acceleration

        self.t += dt

    def run(self, dt, number_of_iterations, number_of_saves=0):
        if not self.initialised:
            self.initialise()

        checkpoints = [i * number_of_iterations / 10 for i in range(10)]  # for progress bar

        if number_of_saves == 0:
            for i in range(number_of_iterations):
                if i in checkpoints:  # progress bar
                    print(str(checkpoints.index(i) * 10) + '% completed')

                self.time_update(dt)

        if number_of_saves > 0:
            iterations_per_save = number_of_iterations // number_of_saves

            saves = [[dt / SECONDS_IN_YEAR * n, []] for n in range(number_of_saves)]

            for i in range(number_of_iterations):
                if i in checkpoints:  # progress bar
                    print(str(checkpoints.index(i) * 10) + '% completed')

                if i % iterations_per_save == 0:
                    saves[i // iterations_per_save][1] = self.point_mass_objects
                self.time_update(dt)

        if 'saves' in locals():
            return ModelSaves(saves)


class ModelSaves:
    def __init__(self, saves):
        self.saves = saves
        self.time_value_saves = [snapshot[0] for snapshot in self.saves]
        self.point_mass_objects_saves = [snapshot[1] for snapshot in self.saves]

    def save_object(self, filename):
        with open(filename, 'wb') as outp:
            pickle.dump(ModelSaves, outp, pickle.HIGHEST_PROTOCOL)

    def show_states(self):
        if len(self.saves) == 0:
            raise ValueError("Trying to plot no saves")

        fig = plt.figure(figsize=[8, 8])

        fax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        sax = fig.add_axes([0.2, 0.9, 0.7, 0.1])

        s_t = Slider(ax=sax, label='data number', valmin=0, valmax=len(self.saves) - 1, valinit=0)

        x_values = np.array([[object.position[0] for object in snapshot] for snapshot in self.saves]).flatten()
        y_values = np.array([[object.position[1] for object in snapshot] for snapshot in self.saves]).flatten()
        minx = np.min(x_values)
        maxx = np.max(x_values)
        miny = np.min(y_values)
        maxy = np.max(y_values)
        maxval = max(maxx, maxy)
        minval = min(minx, miny)

        def update(val):
            x_positions = [i.position[0] for i in self.saves[int(s_t.val)]]
            y_positions = [i.position[1] for i in self.saves[int(s_t.val)]]

            fax.cla()
            fax.scatter(x_positions, y_positions)

            fax.set_title(
                self.saves[int(s_t.val)][1].position[0] ** 2 + self.saves[int(s_t.val)][1].position[1] ** 2)

            k = 0.2  # thickness border
            fax.set_xlim(minval - k * (maxval - minval), maxval + k * (maxval - minval))
            fax.set_ylim(minval - k * (maxval - minval), maxval + k * (maxval - minval))
            fax.set_aspect('equal', adjustable='box')

        s_t.on_changed(update)
        update(0)
        plt.show()

    def plot_path(self):
        for i in range(len(self.point_mass_objects)):
            plt.plot([snapshot[i].position[0] for snapshot in self.saves],
                     [snapshot[i].position[1] for snapshot in self.saves])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def plot_radii(self):
        for i in range(len(self.point_mass_objects)):
            plt.plot([snapshot[i].position[0] * snapshot[i].position[0] + snapshot[i].position[1] *
                      snapshot[i].position[1] for snapshot in self.saves])
        plt.show()

    def plot_resonance_ratio(self, targets):
        target_1 = targets[0]
        target_2 = targets[1]

        semi_major_1 = np.array([snapshot[target_1].calculate_semi_major() for snapshot in self.saves])
        semi_major_2 = np.array([snapshot[target_2].calculate_semi_major() for snapshot in self.saves])

        plt.plot((semi_major_2 / semi_major_1) ** (3 / 2))
        axes = plt.gca()
        axes.set_ybound(lower=0)
        plt.show()

    def plot_semi_major(self, targets=None):
        if targets is None:
            targets = range(len(self.point_mass_objects))

        for i in targets:
            eccentricities = [snapshot[i].calculate_semi_major() for snapshot in self.saves]
            plt.plot(eccentricities, label='planet ' + str(i))
        axes = plt.gca()
        axes.set_ybound(lower=0)
        plt.legend()
        plt.show()

    def plot_eccentricities(self, targets=None):
        if targets is None:
            targets = range(1, len(self.point_mass_objects_saves[0]))

        for i in targets:
            eccentricities = [snapshot[i].calculate_eccentricity() for snapshot in self.point_mass_objects_saves]
            plt.plot(self.time_value_saves, eccentricities, marker='o', label='planet ' + str(i))
        axes = plt.gca()
        axes.set_ybound(lower=0)
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
        return np.cross(self.velocity, self.position)

    def calculate_semi_major(self):
        return -G_TIMES_MASS_SUN / (2 * self.calculate_energy_density())

    def calculate_eccentricity(self):
        l = self.calculate_angular_momentum_density()
        l_squared = np.dot(l, l)
        a = self.calculate_semi_major()
        return np.sqrt(1 - l_squared / (G_TIMES_MASS_SUN * a))


def build_resonance_chain(resonances, eccentricities=None, angles=None, distance=DISTANCE_JUPITER, masses=None):
    N = len(resonances)

    if eccentricities == None:
        eccentricities = np.random.uniform(0., .1, size=(N + 1))
    if angles == None:
        angles = np.random.uniform(0., 2 * np.pi, size=(N + 1))
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


def main():
    dt = 2e6
    N = 10 ** 5

    e = 0.05
    objects = build_resonance_chain([2.], masses=[MASS_JUPITER, MASS_SATURN], eccentricities=[e, 0])

    model = GravityModel(objects)

    saves = model.run(dt, N, 20)

    saves.save_object('Short Eccentricity Test')


if __name__ == "__main__":
    main()

# test for resonance 2 particles
# e = 0.
# meet_angle = 2*np.pi * 0
# distance = distance_jupiter
# objects = build_resonance_chain([2], eccentricities=[e, 0], angles=[0, meet_angle / 2], distance=distance)
