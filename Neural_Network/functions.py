import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

x = np.arange(-20.0, 20.0, 0.1)
y = sigmoid(x)
z = relu(x)
plt.plot(x,y, label="sigmoid")
plt.plot(x,z, linestyle="--", label="reLU")
plt.ylim(-0.3,20.3)
plt.legend()
plt.show()

