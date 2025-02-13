"""
Based on the four features: sepal width, sepal length, petal width and petal length,
we are going to predict the type of flower. Either versicolor or setosa
"""
import numpy as np
import pandas as pd

dataset = pd.read_excel("versicolor-setosa-dataset.xlsx", sheet_name="training")

X = np.array(dataset[['sepal length', 'sepal width', 'petal length', 'petal width']])
y = [1 if label == "Iris-setosa" else -1 for label in dataset["label"]]

weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.01
epochs = 5000

def step_function(z):
    return 1 if z >= 0 else -1

for epoch in range(epochs):
    errors = 0
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) * bias
        prediction = step_function(linear_output)
        error = y[i] - prediction
        if error != 0:
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
            errors += 1
    if errors == 0:
        print("Model converged at epochs:", epoch)
        break

def predict(flower_features):
    output = np.dot(flower_features, weights) + bias
    return step_function(output)


test_dataset = pd.read_excel("versicolor-setosa-dataset.xlsx", sheet_name="test")
test_dataset = np.array(dataset[['sepal length', 'sepal width', 'petal length', 'petal width']])

for i in range(len(test_dataset)):
    result = predict(test_dataset[i])
    print("Prediction for dataset:", i, "Iris-setosa" if (result == 1) else "Iris-versicolor")


