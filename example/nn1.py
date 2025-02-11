import sys
import os
import numpy as np


# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from Tensor import *
import time


# Generate synthetic regression data
def generate_data(n_samples=100):
    X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    y = 0.5 * np.sin(X) + 0.1 * X + np.random.normal(0, 0.1, size=X.shape)
    return X, y


# Create small network for regression
def create_model():
    # Input layer -> Hidden layer (32 neurons)
    w0 = PyTensor(shape=(32, 1), _track_gradient=True, kaimingHeInit=True)
    b0 = PyTensor([[0] * 32], (32, 1), _track_gradient=True)
    
    # Hidden layer -> Hidden layer (16 neurons)
    w1 = PyTensor(shape=(16, 32), _track_gradient=True, kaimingHeInit=True)
    b1 = PyTensor([[0] * 16], (16, 1), _track_gradient=True)

    # Hidden layer -> Output layer (1 neuron)
    w2 = PyTensor(shape=(1, 16), _track_gradient=True, kaimingHeInit=True)
    b2 = PyTensor([[0]], (1, 1), _track_gradient=True)
    
    return w0, b0, w1, b1, w2, b2


def train(X_tensors, y_tensors, w0, b0, w1, b1, w2, b2, epochs=100, lr=0.01):
    n_samples = len(X_tensors)
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(n_samples):
            # Forward pass

            # input layer
            a0 = ((w0 @ X_tensors[i]) + b0).tanh()  # Using tanh activation

            # hidden layer
            a1 = ((w1 @ a0) + b1).tanh()

            # output layer
            y_pred = ((w2 @ a1) + b2)
            
            # Calculate loss
            loss = y_pred.l2(y_tensors[i])
            total_loss += loss.getValue()[0][0]
            
            # Backward pass
            loss.backward()
            
            # Update weights
            w0.sgd(lr=lr)
            b0.sgd(lr=lr)
            w1.sgd(lr=lr)
            b1.sgd(lr=lr)
            w2.sgd(lr=lr)
            b2.sgd(lr=lr)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Average Loss: {total_loss/n_samples:.6f}")


if __name__ == '__main__':
    # Generate data
    X, y = generate_data(n_samples=100)
    
    # Convert to PyTensors
    X_tensors = [PyTensor(values=x.reshape(1, 1), shape=(1, 1), _track_gradient=True) for x in X]
    y_tensors = [PyTensor(values=y_i.reshape(1, 1), shape=(1, 1), _track_gradient=True) for y_i in y]
    
    # Create model
    w0, b0, w1, b1, w2, b2 = create_model()
    
    # Train model
    print("Starting training...")
    t1 = time.time()
    train(X_tensors, y_tensors, w0, b0, w1, b1, w2, b2, epochs=200, lr=0.001)
    training_time = time.time() - t1
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Test predictions
    print("\nTesting predictions...")
    test_x = np.array([-4.0, 0.0, 4.0])
    for x in test_x:
        x_tensor = PyTensor(values=[[x]], shape=(1, 1), _track_gradient=True)
        h0 = ((w0 @ x_tensor) + b0).tanh()
        y_pred = (w1 @ h0) + b1
        print(f"Input: {x:.1f}, Predicted: {y_pred.getValue()[0][0]:.3f}")