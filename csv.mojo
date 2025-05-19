from collections import List, Dict
from sys import simdwidthof

from memory import memset_zero, stack_allocation
from sys import info
from random import rand, randint
from memory.unsafe_pointer import UnsafePointer

#SQRT
from memory import bitcast

from split import train_test_split

alias type = DType.float32

# Get optimal number of elements to run with vectorize at compile time.
# 2x or 4x helps with pipelining and running multiple SIMD operations in parallel.
alias nelts = get_simd_width()

fn get_simd_width() -> Int:
    @parameter
    if info.is_apple_silicon():
        return 4 * simdwidthof[type]()
    else:
        return 2 * simdwidthof[type]()


struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[Scalar[type]]

    # Initialize zeroing all values
    fn __init__(mut self):
        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    #Initializes with random values
    @staticmethod
    fn rand() -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    @staticmethod
    fn randint() -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        randint(data, rows * cols, 0, 5)
        return Self(data)

    # Initialize taking a pointer, don't set any elements
    @implicit
    fn __init__(mut self, data: UnsafePointer[Scalar[type]]):
        self.data = data

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load(y, x)

    fn __setitem__(mut self, y: Int, x: Int, val: Scalar[type]):
        self.store(y, x, val)

    fn load[nelts: Int = 1](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int = 1](self, y: Int, x: Int, val: SIMD[type, nelts]):
        self.data.store(y * self.cols + x, val)

    
    fn pprint(self):
        print("Matrix size:", self.rows,"x", self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                print(self[i,j], end=" ")
            print()
        print()
    
    fn sprint(self, nrows:Int = 5, head: Bool = False):
        if head:
            rstart = 0
            rend = nrows
        else:
            rstart = self.rows - nrows
            rend = self.rows
        print("Matrix size:", self.rows,"x", self.cols)
        print("Showing:", nrows, "rows:")
        for i in range(rstart, rend):
            for j in range(self.cols):
                print(self[i,j], end=" ")
            print()
        print()
        
    fn check_validity(self):
        valid = 0
        validbool = 0
        for i in range(self.rows):
            validbool = 0
            for j in range(self.cols):
                if self[i,j]: validbool = 1
            if validbool == 1: valid += 1

        print("Rows with valid data:", valid)





@parameter
def load_data64(skipheader: Bool = False, data: String = "data.csv") -> Tuple[List[Float64], Int, Int]:
    # Reads data from any csv, defaults to "data.csv". Can skip the first line (headers) with skipheader bool
    dataraw = List[Float64]()
    cols = 0
    rows = 0

    with open(data, 'rb') as file:
        fullstring = file.read()
        stringlist = fullstring.splitlines(keepends=False)
        if skipheader: stringlist = stringlist[1:]
        for i in range(len(stringlist)):
            curstring = stringlist[i]
            values = curstring.split(',')
            for j in range(len(values)):
                # Loads pointer as string -> strips -> floats
                z = Float64(values[j])
                dataraw.append(z)
            if cols==0:
                cols = len(values)
                rows = len(stringlist)
    
    #TODO: Implement split.mojo into this code somehow

    return dataraw, cols, rows #NOTE: FIND SOME WAY TO DYNAMICALLY ALLOCATE MATRICES -> AND RETURN COL AND ROW HERE TOO




@parameter
def load_data(skipheader: Bool = False, data: String = "data.csv") -> Tuple[List[Float32], Int, Int]:
    # Reads data from any csv, defaults to "data.csv". Can skip the first line (headers) with skipheader bool
    dataraw = List[Float32]()
    cols = 0
    rows = 0

    with open(data, 'rb') as file:
        fullstring = file.read()
        stringlist = fullstring.splitlines(keepends=False)
        if skipheader: stringlist = stringlist[1:]
        for i in range(len(stringlist)):
            curstring = stringlist[i]
            values = curstring.split(',')
            for j in range(len(values)):
                # Loads pointer as string -> strips -> floats -> casts as float32
                z = Float64(values[j]).cast[DType.float32]()
                dataraw.append(z)
            if cols==0:
                cols = len(values)
                rows = len(stringlist)
    
    #TODO: Implement split.mojo into this code somehow

    return dataraw, cols, rows #NOTE: FIND SOME WAY TO DYNAMICALLY ALLOCATE MATRICES -> AND RETURN COL AND ROW HERE TOO



#70000 x 784 for MNIST
#150 x 4 for IRIS

def main():
    # NOTE: Known BUG - cannot dynamically allocate struct sizes - hard coded matrix for now.
    dataraw, cols, rows = load_data(data="mnist.csv")
    featuresraw, _, __ = load_data(data='mnistclasses.csv')

    X_train, X_test, y_train, y_test = train_test_split(X=dataraw, y=featuresraw, nfeatures=cols, train_split=0.7)

    training = Matrix[49000, 784]() # HARDCODE IN HERE
    for i in range(training.rows):
        for j in range(training.cols):
            idx = i*cols + j    
            training[i,j] = X_train[idx]

    testing = Matrix[21000, 784]()
    for i in range(testing.rows):
        for j in range(testing.cols):
            idx = i*cols + j    
            testing[i,j] = X_test[idx]

    ytraining = Matrix[49000, 1]()
    for i in range(ytraining.rows):
        for j in range(ytraining.cols):
            idx = i*1 + j    
            ytraining[i,j] = y_train[idx]

    ytesting = Matrix[21000, 1]()
    for i in range(ytesting.rows):
        for j in range(ytesting.cols):
            idx = i*1 + j    
            ytesting[i,j] = y_test[idx]


    training.check_validity()
    testing.check_validity()
