import torch
import torchLt


import numpy as np
import time

import torchLt.Layer
import torchLt.Model
import torchLt.Optimizer


"""Change hyperparameters to test different scenarios"""
# Remark: torchLt so far only supports sizes where the largest parameter fits into vRAM
DATASIZE = 100
EPOCHS = 10

INPUTL = 1
HIDDENL1 = 100000
HIDDENL2 = 1000
OUTPUTL = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torchLt_model = torchLt.Model.Sequential(torchLt.Layer.Linear(INPUTL, HIDDENL1), torchLt.Layer.Tanh(), torchLt.Layer.Linear(HIDDENL1, HIDDENL2), torchLt.Layer.Tanh(), torchLt.Layer.Linear(HIDDENL2, OUTPUTL))
pytorch_model = torch.nn.Sequential(torch.nn.Linear(INPUTL, HIDDENL1), torch.nn.Tanh(), torch.nn.Linear(HIDDENL1, HIDDENL2), torch.nn.Tanh(), torch.nn.Linear(HIDDENL2, OUTPUTL)).to(device)


#torchLt_optimizer = torchLt.Optimizer.Adam(torchLt_model.getParams())
#pytorch_optimizer = torch.optim.Adam(pytorch_model.parameters())

torchLt_optimizer = torchLt.Optimizer.RMSProp(torchLt_model.getParams())
pytorch_optimizer = torch.optim.RMSprop(pytorch_model.parameters())

#torchLt_optimizer = torchLt.Optimizer.Momentum(torchLt_model.getParams())
#pytorch_optimizer = torch.optim.SGD(pytorch_model.parameters(), momentum=0.9)

#torchLt_optimizer = torchLt.Optimizer.SGD(torchLt_model.getParams())
#pytorch_optimizer = torch.optim.SGD(pytorch_model.parameters())




print(f"Model parameters: {torchLt_model.paramCount() / 1000000} million")


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
            y_pred = torchLt_model(X_tensors[i])
            
            # Calculate loss
            loss = y_pred.l2(y_tensors[i])
            total_loss += loss.getValue()[0][0]
            
            # Backward pass
            loss.backward()

            # Optimization step
            opimizer.asyncstep()
            
        if (epoch + 1) % 10 == 0:
            print(f"PyTensor Epoch {epoch+1}, Average Loss: {total_loss/n_samples:.6f}")


def train_pytorch(X, y, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            # Forward pass
            outputs = pytorch_model(X[i])
            loss = criterion(outputs, y[i])
            total_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f"PyTorch Epoch {epoch+1}, Average Loss: {total_loss/len(X):.6f}")


if __name__ == '__main__':
    # Generate data
    X, y = generate_data(n_samples=DATASIZE)

    # PyTorch implementation
    print("\n--- PyTorch Training ---")

    X_torch = torch.FloatTensor(X).to(device)
    y_torch = torch.FloatTensor(y).to(device)

    criterion = torch.nn.MSELoss()
    
    # Train PyTorch model
    t2 = time.time()
    train_pytorch(X_torch, y_torch, criterion=criterion, optimizer=pytorch_optimizer, epochs=EPOCHS)
    pytorch_time = time.time() - t2
    
    print("\n--- Timing Results ---")
    print(f"PyTorch training time: {pytorch_time:.2f} seconds")

    # generate new data to avoid cache-advantages
    X, y = generate_data(n_samples=DATASIZE)

    # torchLt implementation
    print("\n--- torchLt Training ---")
    
    X_tensors = [torchLt.Layer.PyTensor(values=x.reshape(1, 1), shape=(1, 1), _track_gradient=True) for x in X]
    y_tensors = [torchLt.Layer.PyTensor(values=y_i.reshape(1, 1), shape=(1, 1), _track_gradient=True) for y_i in y]

    # Train PyTensor model
    t1 = time.time() 
    train_pytensor(X_tensors, y_tensors, torchLt_optimizer, epochs=EPOCHS)
    pytensor_time = time.time() - t1
    
    print("\n--- Timing Results ---")
    print(f"torchLt training time: {pytensor_time:.2f} seconds")