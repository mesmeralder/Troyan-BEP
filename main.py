import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.widgets import Slider
import copy

matplotlib.use('QtAgg')

G = 6.674e-11 #m3 kg-1 s-2
mass_sun = 1.979e30 #kg
G_times_mass_sun = G*mass_sun #m^3 s^-2

mass_jupiter = 1.898e27 #kg
distance_jupiter = 747.89e9 #m
eccentricity_jupiter = 0.049

mass_saturn = 5.683e26 #kg
distance_saturn = 1.4507e12 #m

mass_neptune = 1.024e26 #kg

mass_uranus = 8.681e25 #kg


class gravity_model:
    def __init__(self, objects, dt):
        if isinstance(objects, list):
            self.point_mass_objects = objects
        elif isinstance(objects, point_mass):
            self.point_mass_objects = [objects]
        else:
            raise Exception("Objects invalid type")

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

    def initialise(self, sun_present=False):
        if not sun_present:
            r_cm = np.array([0.,0.])
            v_cm = np.array([0.,0.])
            for object in self.point_mass_objects:
                r_cm += object.mass * object.position
                v_cm += object.mass * object.velocity

            self.point_mass_objects.append(point_mass(-r_cm / mass_sun, -v_cm / mass_sun, mass_sun))

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

    def time_update(self, save=False, count_periods=False):
        if count_periods:
            for i, mass_object in enumerate(self.point_mass_objects):
                old_is_negative = mass_object.position[1] < 0
                mass_object.position += mass_object.velocity * self.dt + 1 / 2 * mass_object.acceleration * self.dt ** 2
                new_is_positive = mass_object.position[1] >= 0
                if old_is_negative and new_is_positive:
                    self.period_saves[i].append(self.t)

        else:
            for mass_object in self.point_mass_objects:
                mass_object.position += mass_object.velocity * self.dt + 1 / 2 * mass_object.acceleration * self.dt ** 2


        #calculate forces brute force
        forces = np.zeros((len(self.point_mass_objects), 2))
        for (i, j) in combinations(range(len(self.point_mass_objects)), 2):
            object_main = self.point_mass_objects[i]
            object_other = self.point_mass_objects[j]
            force = object_main.gravity(object_other)
            forces[i] += force
            forces[j] -= force

        #leapfrog update x and v
        for i in range(len(self.point_mass_objects)):
            mass_object = self.point_mass_objects[i]
            acceleration = forces[i] / mass_object.mass
            mass_object.velocity += 1/2 * (mass_object.acceleration + acceleration) * self.dt
            mass_object.acceleration = acceleration


        self.t += self.dt

        if save:
            self.saves += [copy.deepcopy(self.point_mass_objects)]

    def run(self, number_of_iterations, iterations_per_save=1, sun_present=False, count_periods=False):
        self.initialise(sun_present=sun_present)

        if count_periods:
            self.period_saves = [[0] for _ in range(len(self.point_mass_objects))]

        if iterations_per_save == 0:
            for i in range(number_of_iterations):
                self.time_update(count_periods=count_periods)

        elif iterations_per_save > 0:
            for i in range(number_of_iterations):
                self.time_update(save=(i % iterations_per_save == 0),count_periods=count_periods)

    def show_saves(self):
        if len(self.saves) == 0:
            raise ValueError("Trying to plot no saves")

        fig = plt.figure(figsize=[8,8])

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

            fax.set_title(self.saves[int(s_t.val)][1].position[0]**2 + self.saves[int(s_t.val)][1].position[1]**2)

            k = 0.2 #thickness border
            fax.set_xlim(minval - k * (maxval - minval), maxval + k * (maxval - minval))
            fax.set_ylim(minval - k * (maxval - minval), maxval + k * (maxval - minval))

        s_t.on_changed(update)
        update(0)
        plt.show()

    def plot_path(self):
        for i in range(len(self.point_mass_objects)):
            plt.plot([snapshot[i].position[0] for snapshot in self.saves], [snapshot[i].position[1] for snapshot in self.saves])
        plt.show()

    def plot_radii(self):
        for i in range(len(self.point_mass_objects)):
            plt.plot([snapshot[i].position[0] * snapshot[i].position[0] + snapshot[i].position[1] * snapshot[i].position[1] for snapshot in self.saves])
        plt.show()

    def plotResonanceRatio(self, targets):
        target_1 = targets[0]
        target_2 = targets[1]

        periods_1 = np.array([snapshot[target_1].CalculateSemiMajor()**3/2 for snapshot in self.saves])
        periods_2 = np.array([snapshot[target_2].CalculateSemiMajor()**3/2 for snapshot in self.saves])

        plt.plot(periods_1 / periods_2)
        plt.show()

    def PlotSemiMajor(self, targets=None):
        if targets is None:
            targets = range(len(self.point_mass_objects))

        for i in targets:
            eccentricities = [snapshot[i].CalculateSemiMajor() for snapshot in self.saves]
            plt.plot(eccentricities)
        plt.legend()
        plt.show()

    def PlotEccentricies(self, targets=None):
        if targets is None:
            targets = range(len(self.point_mass_objects))

        for i in targets:
            eccentricities = [snapshot[i].CalculateEccentricity() for snapshot in self.saves]
            plt.plot(eccentricities)
        plt.legend()
        plt.show()


class point_mass:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass

    @classmethod
    def orbit(cls, distance, mass, angle=0, eccentricity=0):
        velocity = np.sqrt(G_times_mass_sun * (1 - eccentricity) / distance)
        return cls(np.array([np.cos(angle) * distance, np.sin(angle) * distance]), np.array([-np.sin(angle) * velocity, np.cos(angle) * velocity]), mass)

    def __str__(self):
        return str("position = " + str(self.position) + ", velocity = " + str(self.velocity) + ",mass = " + str(self.mass))

    def gravity(self, other_object):
        r = other_object.position - self.position
        abs_r = np.linalg.norm(r)
        return G * self.mass * other_object.mass * r / (abs_r * abs_r * abs_r)

    def CalculateEnergyDensity(self):
        position_norm = np.linalg.norm(self.position)
        velocity_norm = np.linalg.norm(self.velocity)
        return - G_times_mass_sun / position_norm + velocity_norm * velocity_norm / 2

    def CalculateAngularMomentumDensity(self):
        return np.cross(self.velocity, self.position)

    def CalculateSemiMajor(self):
        return -G_times_mass_sun / (2 * self.CalculateEnergyDensity())

    def CalculateEccentricity(self):
        l = self.CalculateAngularMomentumDensity()
        l_squared = np.dot(l,l)
        a = self.CalculateSemiMajor()
        return np.sqrt(1 - l_squared / (G_times_mass_sun * a))

class test_particle:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass

    @classmethod
    def orbit(cls, distance, mass, angle=0, eccentricity=0):
        velocity = np.sqrt(G_times_mass_sun * (1 - eccentricity) / distance)
        return cls(np.array([np.cos(angle) * distance, np.sin(angle) * distance]), np.array([-np.sin(angle) * velocity, np.cos(angle) * velocity]), mass)

    def __str__(self):
        return str("position = " + str(self.position) + ", velocity = " + str(self.velocity) + ",mass = " + str(self.mass))

    def gravity(self, mass_objects):
        acceleration = np.array([0.,0.])
        for point_mass in mass_objects:
            distance = self.position - point_mass.position
            absolute_distance = np.linalg.norm(distance)
            acceleration += G * point_mass.mass * distance / (absolute_distance * absolute_distance * absolute_distance)



def build_resonance_chain(resonances, eccentricities=None, angles=None, distance=distance_jupiter):
    N = len(resonances)

    if eccentricities==None:
        eccentricities = np.random.uniform(0., .1, size=(N+1))
    if angles==None:
        angles = np.random.uniform(0., 2*np.pi, size=(N+1))

    if not (N+1 == len(eccentricities) and N+1 == len(angles)):
        raise Exception("Lengths don't match up")

    objects = []
    objects += [point_mass.orbit(distance, mass_jupiter, eccentricity=eccentricities[0], angle=angles[0])]
    for i in range(N):
        distance = (1 + eccentricities[i+1])/(1 + eccentricities[i]) * distance * resonances[i]**(2/3)
        objects += [point_mass.orbit(distance, mass_saturn, eccentricity=eccentricities[i+1], angle=angles[i+1])]

    return objects


def main():
    dt = 1e6
    N = 10**5

    objects = point_mass.orbit(distance_jupiter, mass_jupiter)

    model = gravity_model(objects, dt)

    model.run(N, iterations_per_save=10**2, count_periods=True)
    model.PlotSemiMajor()
    model.PlotEccentricies()


if __name__ == "__main__":
    main()


#test for resonance 2 particles
# e = 0.
# meet_angle = 2*np.pi * 0
# distance = distance_jupiter
# objects = build_resonance_chain([2], eccentricities=[e, 0], angles=[0, meet_angle / 2], distance=distance)