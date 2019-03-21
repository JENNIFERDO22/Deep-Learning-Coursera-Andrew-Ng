import h5py
import numpy as np 
import matplotlib.pyplot as plt
import scipy
from PIL import Image

#load data sets
#train_set_x, test_set_x (of shape (number of examples, 64, 64, 3))
#train_set_y, test_set_y (of shape (1, number of examples))
#classes (of shape(number of labels,))
train_dataset = h5py.File("train_catvnoncat.h5", "r")
test_dataset = h5py.File("test_catvnoncat.h5", "r")

train_set_x = np.array(train_dataset["train_set_x"][:])
test_set_x = np.array(test_dataset["test_set_x"][:])

train_set_y = np.array(train_dataset["train_set_y"][:])
test_set_y = np.array(test_dataset["test_set_y"][:])

classes = np.array(train_dataset["list_classes"][:])

#reshape data set y
train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
classes = classes.reshape((1, classes.shape[0]))
#change dtype of classes from |S7 to <U13
classes = classes.astype('U13')

#show an image in train set
"""
index = 10
plt.imshow(train_set_x[index])
y = np.squeeze(train_set_y[:,index])
label = np.squeeze(classes[:, y])
print("y =", y, "\nLabel:", label)
plt.show()
"""

#reshape data set x to (number of features 64*64*3, number of examples)
#normalize x
m_train = train_set_x.shape[0]
m_test = test_set_x.shape[0]
num_feautures = 64*64*3

train_set_x = train_set_x.reshape(m_train, -1).T 
test_set_x = test_set_x.reshape(m_test, -1).T

train_set_x = train_set_x/255
test_set_x = test_set_x/255

#initialize w, b with 0s
w = np.zeros((num_feautures, 1))
b = 0

#sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#gradient descent
num_iters = 2000
learning_rate = 0.009
costs = []

for i in range(num_iters):
    z = np.dot(w.T, train_set_x) + b
    a = sigmoid(z)
    dz = a - train_set_y
    dw = 1 / m_train * np.dot(train_set_x, dz.T)
    db = 1 / m_train * np.sum(dz)

    w = w - learning_rate*dw
    b = b - learning_rate*db

    if (i%100 == 0):
        cost = - 1 / m_train * np.sum(train_set_y * np.log(a).T + (1-train_set_y) * np.log(1-a).T)
        costs.append(cost)  


costs = np.squeeze(costs)
plt.plot(costs)
plt.xlabel('cost')
plt.ylabel('iterations (per 100)')
plt.title('learning rate = ' + str(learning_rate))
plt.show()

#accuracy
def accuracy(set_x, set_y):
        z = np.dot(w.T, set_x) + b
        a = sigmoid(z)
        a = np.squeeze(a)
        y_predict = (a > 0.5).astype(float)
        accuracy = (1 - np.sum(abs(y_predict - set_y)) / (set_y.size)) * 100
        return accuracy

print("train accuracy: ", accuracy(train_set_x, train_set_y))
print("test accuracy: ", accuracy(test_set_x, test_set_y))

#predict an image
fname = "cat2.jpg"
#image = np.array(ndimage.imread(fname, flatten=False))
image = np.array(plt.imread(fname))
#my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, -1)).T
my_image = np.array(Image.fromarray(image).resize(size=(64, 64))).reshape((1, -1)).T

if (accuracy(my_image, np.array([1])) == 100.0):
        print("picture prediction: Good")
else:
        print("picture prediction: Bad")