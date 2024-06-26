import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib

matplotlib.use('QtAgg')

G = 6.674e-11  # m3 kg-1 s-2
AU = 1.496e+11 #m
MASS_SUN = 1.979e30  # kg
G_TIMES_MASS_SUN = G * MASS_SUN  # m^3 s^-2


def degrade(array, N=1000):
    if len(array) <= 1001:
        return array
    else:
        dt = (len(array) - 1) / N
        return_list = [array[0]] + [0] * (N)
        for i in range(N):
            return_list[i+1] = array[math.floor((i + 1) * dt)]

        return np.array(return_list)

def plot_semi_major(position_saves, velocity_saves, time_saves):
    position_saves = degrade(position_saves).swapaxes(0, 1)
    velocity_saves = degrade(velocity_saves).swapaxes(0, 1)
    time_saves = degrade(time_saves)

    print(np.shape(position_saves))
    targets = range(1, len(position_saves))
    for i in targets:
        r = np.sqrt(np.sum(position_saves[i]**2, 1))
        print(r[1])
        vsquared = np.sum(velocity_saves[i]**2, 1)
        print(vsquared[0])
        E = .5 * vsquared - G_TIMES_MASS_SUN / r
        semi_majors = - G_TIMES_MASS_SUN / (2 * E * AU)
        plt.plot(time_saves, semi_majors, label='planet ' + str(i))

    axes = plt.gca()
    axes.set_ybound(lower=0, upper=40)
    plt.legend()
    plt.xlabel("time (yr)")
    plt.ylabel("semi-major (AU)")
    plt.show()


def show_states(position_saves, velocity_saves, time_saves):
    fig = plt.figure(figsize=[8, 8])

    fax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    sax = fig.add_axes([0.2, 0.9, 0.7, 0.1])

    s_t = matplotlib.widgets.Slider(ax=sax, label='data number', valmin=0, valmax=len(position_saves) - 1, valinit=0)

    maxval = 30 * AU

    def update(val):
        fax.cla()

        time_value = int(s_t.val)
        for i in range(len(position_saves[time_value])):
            # a = calculate_semi_major(position_saves[time_value, i], velocity_saves[time_value, i])
            # e = calculate_eccentricity(position_saves[time_value, i], velocity_saves[time_value, i])
            # varpi = calculate_varpi(position_saves[time_value, i])
            #
            # b = a * np.sqrt(1 - e*e)
            #
            # kakudo = np.linspace(0, 2*np.pi, 100)
            #
            # x_ellipse = [np.cos(varpi) * (a * np.cos(angle) - a * e) - np.sin(varpi) * b * np.sin(angle) for angle in kakudo]
            # y_ellipse = [np.sin(varpi) * (a * np.cos(angle) - a * e) + np.cos(varpi) * b * np.sin(angle) for angle in kakudo]
            #
            # fax.plot(x_ellipse, y_ellipse)
            fax.scatter(position_saves[time_value, i, 0], position_saves[time_value, i, 1])

        k = 1.2  # thickness border
        fax.set_xlim(-k * maxval, k * maxval)
        fax.set_ylim(-k * maxval, k * maxval)
        fax.set_aspect('equal', adjustable='box')

    s_t.on_changed(update)
    update(0)
    plt.show()


def calculate_semi_major(position_values, velocity_values):
    r = np.sqrt(np.sum(position_values**2, -1))
    v_squared = np.sum(velocity_values**2, -1)
    E = G_TIMES_MASS_SUN/r - .5 * v_squared
    return G_TIMES_MASS_SUN / (2 * E)


def calculate_eccentricity(position_values, velocity_values):
    h = np.cross(position_values, velocity_values, -1)
    return np.linalg.norm(np.cross(velocity_values, h, -1) / G_TIMES_MASS_SUN - position_values / np.linalg.norm(position_values, -1))


def calculate_varpi(position_values):
    return np.arctan2(position_values[..., 0], position_values[..., 1])


def main():
    name = 'Test'

    position_saves = np.load('saves/' + name + '_position.npy', allow_pickle=True)
    velocity_saves = np.load('saves/' + name + '_velocity.npy', allow_pickle=True)
    time_saves = np.load('saves/' + name + '_time.npy', allow_pickle=True)

    print(np.shape(position_saves))

    show_states(position_saves, velocity_saves, time_saves)
    plot_semi_major(position_saves, velocity_saves, time_saves)


if __name__ == "__main__":
    main()