from main import *

with open('saves/Test', 'rb') as f:
    save = pickle.load(f)

save.plot_semi_majors()
save.plot_total_energy()