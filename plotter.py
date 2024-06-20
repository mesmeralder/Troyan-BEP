from main import *



with open('saves/Test', 'rb') as f:
    save = pickle.load(f)

    save.plot_distances()
    save.show_states()
    save.plot_total_energy()