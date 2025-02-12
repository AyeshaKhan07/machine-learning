import numpy as np

X = np.array([
    [32, 50],
    [10, 42],
    [40, 45],
    [55, 10],
    [35, 50],
    [25, 20],
    [32, 30],
    [50, 50],
    [35, 35],
    [42, 60],
    [39, 30],
    [20, 30],
    [10, 50]
])

y = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0])

weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.5

def step_function(z):
    return 1 if z >= 0 else -1

epochs = 5000

for epoch in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)
        error = y[i] - prediction
        weights += learning_rate * error * X[i]
        bias += learning_rate * error


def predict(student_features):
    output = np.dot(student_features, weights) + bias
    return step_function(output)

test_data = np.array([
    [45, 10], # Fail
    [45, 45], # Pass
    [49, 30], # Pass
    [12, 50], # Fail
])

for i in range(len(test_data)):
    result = predict(test_data[i])
    print("Prediction for student:", i, "Pass" if (result == 1) else "Fail")
