import time
from python import Python
import random
from math import sqrt

def euclid(x1, x2):
    # KNOWN BUG: Why must dist be an object? cannot convert x1[i]-like operators to floats(or anything other than object)
    # KNOWN BUG: MOJO cannot use math.sqrt on floats, can use python implementation (slower) or approximation (inaccurate)

    var dist: object = 0.0

    for i in range(len(x1)):
        a = x1[i]
        b = x2[i]
        dist = dist + (a-b) ** 2.0

    target = dist*1000000
    tint = int(target)
    return sqrt(tint) / 1000 


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

    for i in range(split_):
        idx = random.randint(0, datalen-1)
        X_train.append(data.pop(idx))
        y_train.append(classes.pop(idx))

    for i in range(iter_train_):
        idx = random.randint(0, datalen-1)
        X_test.append(data.pop(idx))
        y_test.append(classes.pop(idx))

    return X_train, X_test, y_train, y_test





def main():
    # Imports numpy, imports iris, creates mojo list (outside of speedtests)
    np = Python.import_module("numpy")
    data = np.genfromtxt('data.csv',delimiter=',')
    classes = data[1:, -1]
    data = data[1:, :-1]
    classlist = classes.tolist()
    datalist = data.tolist()

    #del data
    #del classes

    start = time.time.now()

    X_train, X_test, y_train, y_test = train_test_split(datalist, classlist, train_split=0.7)

    for i in range(len(X_test)):
        x = distMat(X_train, X_test[i])
        print(x)
        break


    end = time.time.now()
    total = (end - start) / 1_000_000_000

    print("Mojo Test Time:", total, "Secs")
