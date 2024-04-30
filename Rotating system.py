import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('QtAgg')


G = 6.674e-11 #m3 kg-1 s-2
Msun = 1.979e30 #kg
Mjupiter = 1.898e28 #kg
Vjupiter = 13.07e3 #m/s
Rjupiter = 7.78e11 #m

Xjupiter = Msun / (Msun + Mjupiter) * Rjupiter
Xsun = - Mjupiter / (Msun + Mjupiter) * Rjupiter

omega = Vjupiter / Rjupiter

N = 100
l = np.linspace(-2*Rjupiter, 2*Rjupiter, N)
X,Y = np.meshgrid(l, l, indexing='ij')


#calulcate Centrifugal potential
PotentialCentrifugal = - 1/2 * omega**2 * (X**2 + Y**2)

#calculate Sun potential
PotentialSun = -G*Msun / np.sqrt((X)**2 + Y**2)

#calculate Jupiter potential
PotentialJupiter = -G*Mjupiter / np.sqrt((X - Xjupiter)**2 + Y**2)

Potential = PotentialSun + PotentialCentrifugal + PotentialJupiter

cm = matplotlib.colormaps['viridis']

#plot contour
maxvalue = np.max(Potential)
LevelFromPoint = Potential[N*5//8, int(N/4 * 1/2 * np.sqrt(3))]
levels = -np.exp(np.linspace(np.log(-LevelFromPoint), np.log(-maxvalue), 30))
cp = plt.contour(X, Y, Potential, levels=levels)

plt.scatter(l[N*5//8], l[int(N/2 + N/4 * 1/2 * np.sqrt(3))])

plt.xlabel('X')
plt.ylabel('Y')
plt.show()


#plot surface
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, Potential, cmap=cm,
#                        linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()