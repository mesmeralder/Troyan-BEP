from main import *

with open('Test', 'rb') as f:
    save = pickle.load(f)

save.plot_semi_major()
save.show_states()