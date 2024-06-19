import numpy as np

k = np.random.random([5])
wheres = np.where(k < 0.2)
print(wheres[0])