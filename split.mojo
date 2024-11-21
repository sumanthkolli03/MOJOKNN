from math import floor
from random import randint
from collections import List
from sys import simdwidthof

from random import rand

from algorithm import Static2DTileUnitFunc as Tile2DFunc
from algorithm import parallelize, vectorize
from memory import memset_zero, stack_allocation
from python import Python, PythonObject
from sys import info

alias M = 512  # rows of A and C
alias N = 4096  # cols of B and C
alias K = 512  # cols of A and rows of B
alias type = DType.float32

# Get optimal number of elements to run with vectorize at compile time.
# 2x or 4x helps with pipelining and running multiple SIMD operations in parallel.
alias nelts = get_simd_width()




struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[Scalar[type]]

    # Initialize zeroing all values
    fn __init__(inout self):
        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: UnsafePointer[Scalar[type]]):
        self.data = data

    ## Initialize with random values, useless for now
    @staticmethod
    fn rand() -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load(y, x)

    fn __setitem__(inout self, y: Int, x: Int, val: Scalar[type]):
        self.store(y, x, val)

    fn load[nelts: Int = 1](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int = 1](self, y: Int, x: Int, val: SIMD[type, nelts]):
        self.data.store[width=nelts](y * self.cols + x, val)
    
    def pprint(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print(self[i,j], end=" ")
            print()



fn get_simd_width() -> Int:
    @parameter
    if info.is_apple_silicon():
        return 4 * simdwidthof[type]()
    else:
        return 2 * simdwidthof[type]()








def train_test_split(X: Matrix, y: Matrix, train_split) -> List[Matrix, Matrix, Matrix, Matrix]:
    # X -> Matrix NxM, input data
    # y -> Matrix Nx1, input labels
    # train_split -> float from 0.0 - 1.0

    #NOTE: nested lists are not implemented in mojo, so matrices are required

    #data = List(X)
    #classes = List(y)
    #datasize = int(len(data))
    #split_ = floor(int(len(data)) * train_split)

    X_train = Matrix[, ]()
    y_train = Matrix[, 1]()
    X_test = Matrix[, ]()
    y_test = Matrix[, 1]()

    #datalen = len(data)
    #iter_train_ = len(data) - split_

    #for i in range(split_):
        #idx = randint(0, datalen-1)
        #X_train.append(data.pop(idx))
        #y_train.append(classes.pop(idx))

    #for i in range(iter_train_):
        #idx = randint(0, datalen-1)
        #X_test.append(data.pop(idx))
        #y_test.append(classes.pop(idx))

    #return X_train, X_test, y_train, y_test
    #output_ = [-1,-1,-1,-1]
    #return output_


    #np = Python.import_module("numpy")
    #X_train, X_test, y_train, y_test = sk.train_test_split(X,y,train_size = train_split)

    


def main():
    #Creating Dummy Data using Matrix Struct 
    in_ = List(1,2,3,4,5,6,7,8,9,10,11,12)
    iny_ = List(0,0,1,1,2,2)

    data = Matrix[6,2]()
    classes = Matrix[6,1]()

    for i in range(data.rows):
        for j in range(data.cols):
            idx = i*2 + j
            data[i,j] = in_[idx]
        classes[i,0] = iny_[i]

    #pprint data
    data.pprint()
    classes.pprint()
    print()
    #train_test_split(data,classes,train_split=0.34)
    #print(X_train, X_test, y_train, y_test)

