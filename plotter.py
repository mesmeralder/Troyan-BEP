from main import *

with open('Short Eccentricity Test', 'rb') as f:
    save = pickle.load(f)

save.plot_eccentricities()