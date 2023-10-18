import numpy as np
import matplotlib
import matplotlib.pyplot as plt

A = [2,5,6,7,11,16,19,23,22,29,29,35,37,40,46]
b = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#Visualize data
plt.plot(A,b,'ro')

#chance row vector to column vector
A = np.array([A]).T
b = np.array([b]).T

#print(A.shape[0])

# Creat vector 1
ones = np.ones((A.shape[0],1), dtype=np.int8)

A = np.concatenate((A, ones), axis=1)
#print(A)

#Formular
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
print(x)

# Test data to draw
x0 = np.array([1,46]).T
y0 = x0*x[0][0] + x[1][0]
plt.plot(x0,y0)

# Test predict data
x_test = 12
y_test = x_test*x[0][0] + x[1][0]
print(y_test)

plt.show()