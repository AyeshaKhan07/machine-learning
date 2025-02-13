import numpy as np
import pandas as pd

"""
There are two subjects in the dataset, the student is considered pass if
the sum of the both subjects marks are greater than or equal to 70, else
considered fail
"""

dataset = pd.read_excel("student-pass-fail-data.xlsx")

X = np.array(dataset[['Maths', 'Science']].values)
y = [0 if result == "pass" else 1 for result in dataset["Result"]]

weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.01

def step_function(z):
    return 1 if z >= 0 else 0

epochs = 5000

for epoch in range(epochs):
    errors = 0
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)
        error = y[i] - prediction
        if error != 0:
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
            errors += 1
    if errors == 0:
        print("Model converged at epochs:", epoch)
        break

print(weights, bias)
def predict(student_features):
    output = np.dot(student_features, weights) + bias
    return step_function(output)

test_data = np.array([
    [45, 10], # Fail
    [45, 45], # Pass
    [49, 30], # Pass
    [12, 50], # Fail
    [70, 0],  # Pass
    [0, 70],  # Pass
    [69, 0], # Fail
    [0, 69], # Fail
    [0, 0] # Fail
])

for i in range(len(test_data)):
    result = predict(test_data[i])
    print("Prediction for student:", i, "Pass" if (result == 1) else "Fail")