from numpy import linalg as la
import numpy as np 

#sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

x = np.array([1, 2, 3])
print("sigmoid:", sigmoid(x), "\n")

#sigmoid derivative
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

print("sigmoid derivative:", sigmoid_derivative(x), "\n")

#reshape image chanels to vector
def image2vector(image):
    shape = image.shape
    return image.reshape(shape[0]*shape[1]*shape[2],1)

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("image to vector:")
print(image2vector(image), "\n")

#normalize matrix x by row
def normalizeRow(x):
    norm = la.norm(x, axis=1, keepdims=True)
    return x/norm

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRow: ", normalizeRow(x), "\n")

#vector implementation
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

#dot product
print("dot product:", np.dot(x1,x2), "\n")

#outer product
print("outer product:", np.outer(x1,x2), "\n")

#elementwise multiplication
print("elementwise multiplication:", np.multiply(x1,x2), "\n")

#sum of x square
x = np.array([3, 4, 5])

print("sum of x square:", np.dot(x,x), "\n")

