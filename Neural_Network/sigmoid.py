import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-20.0, 20.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.3,1.3)
plt.show()