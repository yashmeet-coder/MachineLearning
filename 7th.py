import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv("Advertising.csv")

# Define features and target variables
X = data[["TV", "Radio", "Newspaper"]].values  # Assuming these are the features
y = data["Sales"].values  # Assuming "Sales" is the target variable

# Convert data to numpy arrays for easier manipulation
# X = np.array(X)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = np.c_[np.ones(X.shape[0]), X]
print(X)
# y = np.array(y)

# Function to calculate predicted sales
def predict(X, theta):
  return np.dot(X, theta)

# Function to calculate the mean squared error (cost function)
def mse(y_true, y_predicted):
  return np.mean((y_true - y_predicted) ** 2)

# Learning rate (controls how much we update weights each iteration)
learning_rate = 0.01

# Initialize weights and bias with random values
theta = np.zeros(X.shape[1])  # +1 for bias term

# Define the number of iterations
num_iterations = 1000

# Gradient descent loop
for _ in range(num_iterations):
  # Predict sales based on current weights and bias
  y_predicted = predict(X, theta)

  # Calculate the gradient of the cost function
  gradient = -(2/X.shape[0]) * np.dot(X.T, (y_predicted - y))
#   print(gradient)
  # Update weights and bias using the learning rate
  theta -= learning_rate * gradient

# Print the final weights and bias
print(theta)
print("Final weights:", theta[:-1])  # Print weights excluding bias
print("Final bias:", theta[-1])

# Use the trained model to predict sales for new data (optional)
# new_data = ...  # Replace with your new data
# new_predicted_sales = predict(new_data, theta)
# print("Predicted sales for new data:", new_predicted_sales)
