from main import *

with open('Short Eccentricity Test', 'rb') as f:
    save = pickle.load(f)

save.plot_eccentricities()
save.plot_semi_major()
save.show_states()