from main import *


for i in range(10):
    with open('saves/Density_test_2_' + str(i), 'rb') as f:
        save = pickle.load(f)

    save.plot_distances()