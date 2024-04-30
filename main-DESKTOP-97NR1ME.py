import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.widgets import Slider
import copy

matplotlib.use('QtAgg')

G = 6.674e-11 #m3 kg-1 s-2
Msun = 1.979e30 #kg
Mjupiter = 1.898e27 #kg
Vjupiter = 13.07e3 #m/s
Rjupiter = 747.89e9 #m

mu = Mjupiter / (Mjupiter + Msun)
omega = np.sqrt(G * (Msun + Mjupiter) / Rjupiter**3)


class gravity_model:
    def __init__(self, objects, dt):
        self.point_mass_objects = objects
        self.dt = dt
        self.saves = []
        self.t = 0

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
        forces = {mass_object: np.zeros(2) for mass_object in self.point_mass_objects}
        for (object_main, object_other) in combinations(self.point_mass_objects, 2):
            force = object_main.gravity(object_other)
            forces[object_main] += force
            forces[object_other] -= force

        for mass_object in self.point_mass_objects:
            mass_object.acceleration = forces[mass_object] / mass_object.mass

    def time_update(self, save=False):
        #calculate forces brute force
        forces = {mass_object : np.zeros(2) for mass_object in self.point_mass_objects}
        for (object_main, object_other) in combinations(self.point_mass_objects, 2):
            force = object_main.gravity(object_other)
            forces[object_main] += force
            forces[object_other] -= force

        #leapfrog update x and v
        for mass_object in self.point_mass_objects:
            acceleration = forces[mass_object] / mass_object.mass
            mass_object.position += mass_object.velocity * self.dt + 1/2 * acceleration * self.dt**2
            mass_object.velocity += 1/2 * (mass_object.acceleration + acceleration) * self.dt
            mass_object.acceleration = acceleration


        self.t += 1

        if save:
            self.saves += [copy.deepcopy(self.point_mass_objects)]

    def run(self, number_of_iterations, iterations_per_save=1):
        self.initialise()

        if iterations_per_save == 0:
            for i in range(number_of_iterations):
                self.time_update()

        elif iterations_per_save > 0:
            for i in range(number_of_iterations):
                self.time_update(save=(i % iterations_per_save == 0))

    def plot_saves(self):
        if len(self.saves) == 0:
            raise ValueError("Trying to plot no saves")

        fig = plt.figure(figsize=[8,8])

        fax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        sax = fig.add_axes([0.2, 0.9, 0.7, 0.1])

        s_t = Slider(ax=sax, label='data number', valmin=0, valmax=len(self.saves) - 1, valinit=0)

        x_values = np.array([[object.position[0] for object in snapshot] for snapshot in self.saves]).flatten()
        y_values = np.array([[object.position[0] for object in snapshot] for snapshot in self.saves]).flatten()
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

            fax.set_title(self.saves[int(s_t.val)][1].position[0]**2 + self.saves[int(s_t.val)][1].position[1]**2)

            k = 0.2 #thickness border
            fax.set_xlim(minval - k * (maxval - minval), maxval + k * (maxval - minval))
            fax.set_ylim(minval - k * (maxval - minval), maxval + k * (maxval - minval))

        s_t.on_changed(update)
        update(0)
        plt.show()

    def plot_path(self, object_numbers):
        for i in object_numbers:
            plt.plot([snapshot[i].position[0] for snapshot in self.saves], [snapshot[i].position[1] for snapshot in self.saves])
        plt.show()


class point_mass:
    def __init__(self, position, velocity, m):
        self.position = position
        self.velocity = velocity
        self.mass = m

    def __str__(self):
        return str("position = " + str(self.position) + ", velocity = " + str(self.velocity) + ",mass = " + str(self.mass))

    def gravity(self, other_object):
        r = other_object.position - self.position
        abs_r = np.linalg.norm(r)
        return G * self.mass * other_object.mass * r / (abs_r ** 3)


def main():
    objects = []

    #chaotic orbit
    # objects += [point_mass(np.array([0.,0.]), np.array([0.,0.]), Msun)]
    # objects += [point_mass(np.array([Rjupiter,0.]), np.array([0.,Vjupiter]), Mjupiter)]
    # objects += [point_mass(np.array([1/2 * Rjupiter,1/2 * np.sqrt(3) * Rjupiter]), 1.1 * np.array([- 1/2 * np.sqrt(3) * Vjupiter, 1/2 * Vjupiter]), Mjupiter / 1000)]

    #stable L4
    objects += [point_mass(np.array([-mu * Rjupiter,0.]), np.array([0.,-mu * Vjupiter]), Msun)]
    objects += [point_mass(np.array([(1 - mu) * Rjupiter,0.]), np.array([0.,(1 - mu) * Vjupiter]), Mjupiter)]
    #objects += [point_mass(np.array([1/2 * Rjupiter,1/2 * np.sqrt(3) * Rjupiter]), np.array([- 1/2 * np.sqrt(3) * Vjupiter, 1/2 * Vjupiter]), Mjupiter / 1000)]

    

    model = gravity_model(objects, 1e5)

    model.run(10**5)
    model.plot_saves()


if __name__ == "__main__":
    main()