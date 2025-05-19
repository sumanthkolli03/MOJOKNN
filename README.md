# Mojo-based KNN

## Contact:
* _Email: sumanthkolli03@gmail.com_
* _Email: signaclee@gmail.com_


Run in mojo 25.1.1
`Copy necessary files into your magic environment, and run as needed. Note that files may not be in the same directories as needed to run.`


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
* QOL: better outputting and data loading
Note that you may have to change some of the hardcoded lines like data and training sizes. We would like this to be parametric, or possible to enter easily, but unfortunately bugs in the MOJO compiler prevent this from working. UPDATE 5/19/25: This issue has been recognized by the mojo devs, and they are working on it.

### /Tests
Contains slightly outdated tests primarily used for speedtesting. Try them out yourself!
These are the same tests present in our paper: <insert link>

### /Data
Contains code used to download all the datasets. Try it out yourself!
