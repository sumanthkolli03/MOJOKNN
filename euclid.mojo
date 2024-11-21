from math import floor, sqrt
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
        print("Matrix size:", self.rows,"x", self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                print(self[i,j], end=" ")
            print()
        print()



fn get_simd_width() -> Int:
    @parameter
    if info.is_apple_silicon():
        return 4 * simdwidthof[type]()
    else:
        return 2 * simdwidthof[type]()

def euclid(x1: Matrix, x2: Matrix):
    # KNOWN BUG: MOJO cannot use math.sqrt on floats, must use python implementation (slower + more bugs) or approximation (inaccurate)
    # KNOWN BUG: Can't declare return type as Float32 (even though thats what it is.) Might require explicit type switching from Float64->Float32 (even though it isnt a Float64)

    # x1: matrix slice, one point of original training data, 1 x m
    # x2: matrix slice, one point of testing data, 1 x m
    # returns euclidean distance (sqrt of square of distances)

    var dist: Float32 = 0.0

    for j in range(x1.cols):
        a = x1[0, j]
        b = x2[0, j]
        dist = dist + (a-b) ** 2.0

    target = dist*1000000
    tint = int(target)
    return sqrt(tint) / 1000 




def main():
    #Creating Dummy Data using Matrix Struct 
    in_ = List(3,4,2)
    in2_ = List(0,0,0)

    data = Matrix[1, 3]()
    data2 = Matrix[1, 3]()

    for i in range(data.rows):
        for j in range(data.cols):
            idx = i*2 + j
            data[i,j] = in_[idx]
            data2[i,j] = in2_[idx]

    #pprint data
    data.pprint()
    data2.pprint()

    print(euclid(data, data2))
