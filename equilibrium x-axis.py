import numpy as np
import matplotlib.pyplot as plt
import scipy

k = 1/2 #mass ratio
distance = 5

x_1 = - 1/(1 + 1/k) * distance
x_2 = 1/(1 + k) * distance

def f(x):
    x_01 = x_1 - x
    x_02 = x_2 - x
    r_01 = abs(x_01)
    r_02 = abs(x_02)
    return k * x_01 / r_01**3 + x_02 / r_02**3 + (1 + k) / distance ** 3 * x

x_values = np.linspace(-2*distance, 2 * distance, 100)
y_values = f(x_values)

x_min = x_values[x_values < x_1]
x_mid = x_values[np.array([x > x_1 and x < x_2 for x in x_values])]
x_max = x_values[x_values > x_2]

y_min = f(x_min)
y_mid = f(x_mid)
y_max = f(x_max)

plt.figure()
plt.plot(x_min, y_min, c="mediumseagreen")
plt.plot(x_mid, y_mid, c="mediumseagreen")
plt.plot(x_max, y_max, c="mediumseagreen")

sol_min = scipy.optimize.root(f, 2*x_1).x
sol_mid = scipy.optimize.root(f, 0).x
sol_max = scipy.optimize.root(f, 2*x_2).x
x_solutions = np.array([sol_min, sol_mid, sol_max])
y_solutions = f(x_solutions)

plt.plot(x_solutions, y_solutions, 'ro')

plt.plot()
lines = plt.vlines([x_1, x_2], -1, 1, colors='darkmagenta')

ax = plt.gca()
ax.set_ylim([-1, 1])

plt.xticks([])
plt.yticks([])

plt.xlabel("distance")
plt.ylabel("reduced force (kg m^-3)")

plt.show()