import numpy as np

R=0
X=np.array([[1,0,1],[1,1,1],[1,1,-1],[-1,1,1]])
Y = [2,2.7,-0.7,2]
theta = np.array([0,1,2])
for i in range(4):
    z = Y[i] - np.inner(X[i],theta)
    R += (z**2)/2
R = R/4
print('Risk is : ', R)