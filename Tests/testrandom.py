from sklearn.neighbors import KNeighborsClassifier
from time import perf_counter_ns
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def skKNN(train_data, train_labels, test_data, k):
    KNN = KNeighborsClassifier(n_neighbors = k, algorithm="brute")
    KNN.fit(train_data, train_labels)
    predictions = KNN.predict(test_data)
    return predictions

if __name__ == "__main__":

    training = np.random.rand(250000, 400)
    testing = np.random.rand(1000, 400)
    trainingclasses = np.random.randint(4, size=250000)

    start_time = perf_counter_ns()
    preds = skKNN(training, trainingclasses, testing, 100)
    end_time = perf_counter_ns()
    elapsed_time = end_time - start_time
    
    print("Execution time: ", (elapsed_time/1000000000) , "seconds or ", elapsed_time, "nanoseconds")
    print(f"{(elapsed_time/1000000000)}, ")