import sys
import os


# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


from torchLt import *


from torchLt.Layer import Linear, Tanh
from torchLt.Model import Sequential
from torchLt.Optimizer import SGD
import numpy as np
import time


model = Sequential(Linear(1, 100000), Tanh(), Linear(100000, 1000), Tanh(), Linear(1000, 1))

print(f"Model parameters: {model.paramCount() / 1000000} million")


# Generate synthetic regression data
def generate_data(n_samples=1000):
    X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    y = 0.5 * np.sin(X) + 0.1 * X + np.random.normal(0, 0.1, size=X.shape)
    return X, y

# PyTensor Implementation
def train_pytensor(X_tensors, y_tensors, opimizer, epochs=100):
    n_samples = len(X_tensors)
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(n_samples):
            y_pred = model(X_tensors[i])
            
            # Calculate loss
            loss = y_pred.l2(y_tensors[i])
            total_loss += loss.getValue()[0][0]
            
            # Backward pass and optimization
            loss.backward()

            # Optimization step
            opimizer.asyncstep()
            
        if (epoch + 1) % 10 == 0:
            print(f"PyTensor Epoch {epoch+1}, Average Loss: {total_loss/n_samples:.6f}")



if __name__ == '__main__':
    # Generate data
    X, y = generate_data(n_samples=100)
    
    print("\n--- PyTensor Training ---")
    # PyTensor implementation
    X_tensors = [PyTensor(values=x.reshape(1, 1), shape=(1, 1), _track_gradient=True) for x in X]
    y_tensors = [PyTensor(values=y_i.reshape(1, 1), shape=(1, 1), _track_gradient=True) for y_i in y]
    
    # Train PyTensor model
    t1 = time.time()
    train_pytensor(X_tensors, y_tensors, SGD(model.getParams(), lr=0.001), epochs=10)
    pytensor_time = time.time() - t1
    
    print("\n--- Timing Results ---")
    print(f"PyTensor training time: {pytensor_time:.2f} seconds")