import numpy as np

#if문을 사용하여 0,1값 return 가능 하겠지만 numpy배열이 들어가지 못하므로 아래와 같이 구현
def step_function(x):
    y = x > 0 #비교 연산자이기 때문에 y에 bool값이 저장됨
    return y.astype(np.int32) #위의 bool값을 int형으로 바꿔줌

print(step_function(np.array([1.0, -2.0, 3.0])))
print(step_function(np.array([1.0, 2.0])))

