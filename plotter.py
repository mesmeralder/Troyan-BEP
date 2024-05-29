import pickle
from main import ModelSaves

with open('Short Eccentricity Test', 'rb') as f:
    save = pickle.load(f)

print(save.time_value_saves)