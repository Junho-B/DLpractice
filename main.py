import numpy as np

# 편향(bias)를 이용한 퍼셉트론 구현
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

a = AND(1, 2)
print(a)
