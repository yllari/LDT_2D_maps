import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.colors import BoundaryNorm

coords = np.genfromtxt("coords.txt")
value = np.genfromtxt("value.txt")

# Transforming this to array again
div = int(np.sqrt(len(value) ) )
value = value.reshape(div, div)
plt.imshow(value, origin="lower", extent=[0,1,0,1])
#plt.hist(value)
plt.colorbar()
plt.show()
