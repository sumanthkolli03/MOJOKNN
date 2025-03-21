# Mojo-based KNN

## Contact:
* _Email: sumanthkolli03@gmail.com_
* _Email: signaclee@gmail.com_


Run in mojo 25.1.1
Copy necessary files into your magic environment, and run as needed. Note that files may not be in the same directories as needed to run.


### Split.mojo
    Contains the code used to split data into training and testing sets.

### csv.mojo
    Contains the code used to load data into Lists. Note that most of the KNN program uses Matrices, as they are built to use vectorization and SIMD.  
    Also, there is a known bug within mojo that doesn't allow dynamic allocation of fix-length structs. This does unfortunately have to be hard coded to your data size before-hand. 

### KNN.mojo
    Contains the code to run your own KNN! Functionality includes returning all points' predicted classes.
    TODO:
    * Distance from points
    * Different distance metrics (manhattan, cosine, etc.)
    * Different algorithms (Ball-Tree, KD-Tree)

### /Tests
    Contains different tests primarily used for speedtesting. Try them out yourself!

### /Data
    Contains Iris and MNIST-784 data. Note that you may have to unzip MNIST data yourself.