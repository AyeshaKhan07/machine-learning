import numpy as np

# Step 1: Define the dataset
# Features: [Suspicious Keywords, Number of Links, Email Length]
# Labels: 1 = Spam, 0 = Non-Spam
X = np.array([
    [3, 5, 100],  # Example 1: Spam
    [1, 0, 50],   # Example 2: Non-Spam
    [4, 7, 150],  # Example 3: Spam
    [0, 0, 20],   # Example 4: Non-Spam
    [2, 3, 70]    # Example 5: Spam
])

y = np.array([1, 0, 1, 0, 1])  # Corresponding labels

# Step 2: Initialize weights and bias
weights = np.zeros(X.shape[1])  # Initialize weights to zero
bias = 0
learning_rate = 0.01  # Small learning rate for gradual updates

# Step 3: Define the activation function (Step function)
def step_function(z):
    return 1 if z >= 0 else 0

# Step 4: Train the perceptron
epochs = 50  # Number of iterations over the dataset

for epoch in range(epochs):
    for i in range(len(X)):
        print("i", i)
        # Calculate the perceptron's output
        linear_output = np.dot(X[i], weights) + bias
        print("linear_output", linear_output)
        prediction = step_function(linear_output)
        
        # Compute the error
        error = y[i] - prediction
        
        # Update weights and bias
        print("weights", weights)
        weights += learning_rate * error * X[i]
        print("updated weights", weights)
        bias += learning_rate * error

# Step 5: Test the perceptron
def predict(email_features):
    linear_output = np.dot(email_features, weights) + bias
    return step_function(linear_output)

# Example test cases
test_email_1 = [2, 4, 80]  # Likely Spam
test_email_2 = [0, 1, 30]  # Likely Non-Spam

print("Prediction for Test Email 1:", "Spam" if predict(test_email_1) else "Non-Spam")
print("Prediction for Test Email 2:", "Spam" if predict(test_email_2) else "Non-Spam")
