from main import *


with open('saves/resonance accuracy test/Resonance accuracy test, dt=1e6s', 'rb') as f:
    save = pickle.load(f)

    save.plot_distances()
    save.show_states()
    save.plot_total_energy()