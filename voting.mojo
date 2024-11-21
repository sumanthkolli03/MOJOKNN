## TODO: implement vector struct, fix dtype issues (Float64?)
## TODO: implement .append for vector-like structs. Include internal counter of idx and have it add 1 each time its called and update vector[, idx]
## TODO: implement vecor slicing of matrix. For example: Matrix[0, :] -> Vector[m] OR Matrix[:, 0] -> Vector[n]

from collections import List
from collections import Counter
import time
from python import Python, PythonObject

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

    def index(self, value_to_idx: Float32) -> Int:
        # NOTE: Currently a workaround, converts to list and finds index. Needs to be updated to use memory
        a = List[Float32](-100)
        for i in range(self.cols):
            a.append(self[0,i])
        out = a.index(value_to_idx) + 1
        return out



fn get_simd_width() -> Int:
    @parameter
    if info.is_apple_silicon():
        return 4 * simdwidthof[type]()
    else:
        return 2 * simdwidthof[type]()

def count_votes(votes: List):
    #KNOWN BUG: Coutner does not work as intended
    #c = Counter[String](votes)
    #most = c.most_common(1)
    #print(most)
    return -1

def get_min(array_distances: Matrix): 
    # Same bugs as euclid, easiest (and potentially fastest?) to just use numpy for this

    # array_distances -> 1 x M matrix (vector), all the distances from euclid compiled into a vector

    var minimum_val: Float32 = 0.0
    minimum_val = minimum_val + array_distances[0, 0]

    for i in range(1, array_distances.cols): 
        value = array_distances[0, i]
        if (value < minimum_val):
            minimum_val = array_distances[0, i]

    result = array_distances.index(minimum_val)

    return result


def voting(array_distances: Matrix, array_y: Matrix, k: Int) -> Matrix:
    #KNOWN BUG: Can't initialize empty mutable lists

    # array_distances -> n x 1 matrix (vector), all the distances from euclid compiled into a vector
    # array_y -> n x 1 matrix (vector), all the original classes mapped to the training points 
    var votes: Matrix = Matrix[1, array_distances.rows]()
    

    #Note: needs updates to use List popping IDX methods. Should not rewrite matrix everytime, but should rewrite a temp list of IDX's 
    i = 0
    while i<k:

        idx = get_min(array_distances)
        _ = array_distances.pop(idx)
        vote_ = array_y.pop(idx)
        votes.append(vote_)
        i = i+1


    
    _ = votes.pop(0) # cant init empty lists, need to pop initial value

    
    #tallied_votes = count_votes(votes)
    return votes
    #return tallied_votes




def main():


    start = time.time.now()

    #NOTE: These should both be vectors
    var array_distances_ = List[Float32](0.12,3.19,4,5,6,7,8,9,0)
    var train_classes_ = List[Float32](0,1,0,1,0,1,2,2,2)

    array_distances = Matrix[1, 9]()
    train_classes = Matrix[1, 9]()

    for i in range(array_distances.rows):
        for j in range(array_distances.cols):
            idx = i*2 + j
            array_distances[i,j] = array_distances_[idx]
            train_classes[i,j] = train_classes_[idx]

    array_distances.pprint()
    train_classes.pprint()

    #vote_out = voting(array_distances, train_classes, 3)

    end = time.time.now()
    total = (end - start) / 1_000_000_000

    print("Mojo Test Time:", total, "Secs")
