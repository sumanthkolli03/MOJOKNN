import time
import random
import math


def euclid(x1,x2):
    dist = 0.0
    for i in range(len(x1)):
        dist += (x1[i] - x2[i]) ** 2
    return math.sqrt(dist)

def distMat(training_data, input_point):
    distmat = []
    for i in range(len(training_data)):
        distmat.append(
            euclid(training_data[i], input_point)
        )
    return distmat

def train_test_split(X, y, train_split):

    data = X.copy()
    classes = y.copy()

    split_ = math.floor(len(data) * train_split)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    datalen = len(data)
    iter_train_ = len(data) - split_

    for _ in range(split_):
        idx = random.randint(0, datalen-1)
        X_train.append(data.pop(idx))
        y_train.append(classes.pop(idx))
        datalen -= 1

    for _ in range(iter_train_):
        idx = random.randint(0, datalen-1)
        X_test.append(data.pop(idx))
        y_test.append(classes.pop(idx))
        datalen -= 1

    return X_train, X_test, y_train, y_test





def main():
    # Imports numpy, imports iris, creates mojo list (outside of speedtests)
    import numpy as np
    data = np.genfromtxt('data.csv',delimiter=',')
    classes = data[1:, -1]
    data = data[1:, :-1]
    classlist = classes.tolist()
    datalist = data.tolist()

    #del data
    #del classes

    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(datalist, classlist, train_split=0.7)

    for i in range(len(X_test)):
        x = distMat(X_train, X_test[i])
        print(x)
        break

    end = time.time()
    total = (end - start) / 1_000_000_000

    print("Mojo Test Time:", total, "Secs")


if __name__ == "__main__":
    main()