import numpy as np
def sigmoid(X):
    return 1 / (1 + np.exp(-X))
A = np.array([1,2,3,4])
B = np.array([[1,2],[3,4],[5,6]])
print(A)
print(np.ndim(A))
print(np.ndim(B))
print(B.shape)
S = ([[1,2,3],[4,5,6],[7,8,9]])
O = ([[3,4],[2,3],[7,8]])
print(np.dot(S, O))
print("-------------------------")
#neural network에서의 matrix multiplication
X = np.array([3.0,2.0]) #input
W = np.array([[2.0,4.0,6.0],[3.0,6.0,9.0]]) #weight
Y = np.dot(X, W) #output
print(Y)
print("-------------")
X = np.array([1.0,2.0])
W1 = np.array([[1.0,3.0,5.0],[2.0,4.0,6.0]])
B1 = np.array([1.0,2.0,3.0])
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(Z1)
print("-------------")
W2 = np.array([[0.3,0.4],[0.2,0.6],[0.5,0.7]])
B2 = np.array([0.2, 0.5])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(A2)

print("-------------")
W3 = np.array([[0.3, 0.5],[0.7,0.2]])
B3 = np.array([0.2, 0.1])
A3 = np.dot(Z2, W3) + B3
print(A3)


