import numpy as np
import matplotlib.pyplot as plt

weights = np.array([10, 0.2, -0.75])
learning_rate = 0.05

# Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Training data for 2-bit AND gate logic
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Initialize variables for tracking epochs and errors
epochs = 0
errors = []

# Main loop for perceptron learning
while True:
    error = 0
    for i in range(len(X)):
        # Calculate the predicted output
        net_input = np.dot(np.insert(X[i], 0, 1), weights)
        y_pred = step_activation(net_input)
        
        # Update the weights
        delta = learning_rate * (y[i] - y_pred)
        weights += delta * np.insert(X[i], 0, 1)
        
        # Calculate the error for this sample
        error += (y[i] - y_pred) ** 2
    
    # Calculate the sum-squared error for this epoch
    error /= len(X)
    errors.append(error)
 
    epochs += 1
    
    # Check for convergence or maximum iterations
    if error <= 0.002 or epochs >= 1000:
        break

plt.plot(range(1, epochs + 1), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error')
plt.grid(True)
plt.show()

# Print the number of epochs needed for convergence
print(f"Number of epochs for convergence: {epochs}")

print(f"Final weights: W0 = {weights[0]}, W1 = {weights[1]}, W2 = {weights[2]}")

