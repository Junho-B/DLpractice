import numpy as np

def init_neural_network():
    network = {}
    network['W1'] = np.array([[0.3,0.5,0.1],[0.2,0.7,0.8]])
    network['B1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.5,0.8],[0.6,0.3],[0.4,0.2]])
    network['B2'] = np.array([0.2,0.3])
    network['W3'] = np.array([[0.2,0.5],[0.7,0.8]])
    network['B3'] = np.array([0.5,0.6])
    return network

def fw(network,x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['B1'],  network['B2'],  network['B3']
    a1 = np.dot(x,w1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,w2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3)+b3
    z3 = identity_function(a3)
    return z3

def identity_function(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

network = init_neural_network()
x = np.array([0.5,0.9])
result = fw(network,x)
print(result)

