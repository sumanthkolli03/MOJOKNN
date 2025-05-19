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
    data = np.loadtxt('cancer.csv', delimiter=',')
    y = np.loadtxt('cancerClasses.csv', delimiter=',')
    X = data

    training, testing, trainingclasses, y_test = train_test_split(X, y, train_size=0.8)

    start_time = perf_counter_ns()
    preds = skKNN(training, trainingclasses, testing, 10)
    end_time = perf_counter_ns()
    elapsed_time = end_time - start_time
    
    print("Execution time: ", (elapsed_time/1000000000) , "seconds or ", elapsed_time, "nanoseconds")
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    print(f"{(elapsed_time/1000000000)}, ")
