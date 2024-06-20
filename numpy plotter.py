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
        print(np.shape(position_saves[i]))
        r = np.sqrt(np.sum(position_saves[i]**2, 1))
        print(r[0])
        vsquared = np.sum(velocity_saves[i]**2, 1)
        E = .5 * vsquared - G_TIMES_MASS_SUN / r
        semi_majors = G_TIMES_MASS_SUN / (2 * E * AU)
        print(semi_majors[0])
        plt.plot(time_saves, semi_majors, label='planet ' + str(i))

    axes = plt.gca()
    axes.set_ybound(lower=0, upper=40)
    plt.legend()
    plt.xlabel("time (yr)")
    plt.ylabel("semi-major (AU)")
    plt.show()


def main():
    name = 'Test'

    position_saves = np.load('saves/' + name + '_position.npy', allow_pickle=True)
    velocity_saves = np.load('saves/' + name + '_velocity.npy', allow_pickle=True)
    time_saves = np.load('saves/' + name + '_time.npy', allow_pickle=True)

    print(np.shape(position_saves))

    plot_semi_major(position_saves, velocity_saves, time_saves)


if __name__ == "__main__":
    main()