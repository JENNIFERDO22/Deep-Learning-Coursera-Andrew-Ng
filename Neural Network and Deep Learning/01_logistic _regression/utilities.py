import h5py
import numpy as np 
import matplotlib.pyplot as plt

#load data sets
def load_dataset():
    train_dataset = h5py.File("train_catvnoncat.h5", "r")
    test_dataset = h5py.File("test_catvnoncat.h5", "r")

    train_set_x = np.array(train_dataset["train_set_x"][:])
    test_set_x = np.array(test_dataset["test_set_x"][:])

    train_set_y = np.array(train_dataset["train_set_y"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])

    classes = np.array(train_dataset["list_classes"][:])

    #reshape data sets 
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
    classes = classes.reshape((1, classes.shape[0]))
    #change dtype of classes from |S7 to <U13
    classes = classes.astype('U13')

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


#show an image in data set
def show_img(dataset_x, dataset_y, classes, index):
    plt.imshow(dataset_x[index])
    y = np.squeeze(dataset_y[:,index])
    label = np.squeeze(classes[:, y])
    print("y =", y, "\nLabel:", label)
    plt.show()