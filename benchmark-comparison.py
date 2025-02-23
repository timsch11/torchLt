import torch
import torchLt
import numpy as np
import time
from typing import Dict, List

import altair as alt
import pandas as pd

import gc
import math

import torchLt.Layer
import torchLt.Model
import torchLt.Optimizer

def generate_data(n_samples=100):
    X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    y = 0.5 * np.sin(X) + 0.1 * X + np.random.normal(0, 0.1, size=X.shape)
    return X, y


def create_pytorch_model(input_size: int, hidden_size1: int, hidden_size2: int, output_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pytorch_model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size1),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size1, hidden_size2),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size2, output_size)
    ).to(device)
    
    return pytorch_model

def create_torchlt_model(input_size: int, hidden_size1: int, hidden_size2: int, output_size: int):
    
    torchLt_model = torchLt.Model.Sequential(
        torchLt.Layer.Linear(input_size, hidden_size1),
        torchLt.Layer.Tanh(),
        torchLt.Layer.Linear(hidden_size1, hidden_size2),
        torchLt.Layer.Tanh(),
        torchLt.Layer.Linear(hidden_size2, output_size)
    )
    
    return torchLt_model


def get_pytorch_optimizer(model_pytorch, optimizer_name: str):
    if optimizer_name == "SGD":
        return torch.optim.SGD(model_pytorch.parameters())
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model_pytorch.parameters())
    elif optimizer_name == "RMSProp":
        return torch.optim.RMSprop(model_pytorch.parameters())
    elif optimizer_name == "Momentum":
        return torch.optim.SGD(model_pytorch.parameters(), momentum=0.9)
    

def get_torchlt_optimizer(model_torchLt, optimizer_name: str):
    if optimizer_name == "SGD":
        return torchLt.Optimizer.SGD(model_torchLt.getParams())
    elif optimizer_name == "Adam":
        return torchLt.Optimizer.Adam(model_torchLt.getParams())
    elif optimizer_name == "RMSProp":
        return torchLt.Optimizer.RMSProp(model_torchLt.getParams())
    elif optimizer_name == "Momentum":
        return torchLt.Optimizer.Momentum(model_torchLt.getParams())


def benchmark_training(param_size: int, optimizer_name: str, epochs: int = 5, data_size: int = 100) -> tuple:
    # Calculate hidden layer size to achieve target parameter count
    # Parameters = (input_size * hidden_size + hidden_size) + (hidden_size * output_size + output_size)
    input_size = 1
    output_size = 1

    # weight matrix between hidden layers = param_size -> actual param size = param_size + 4*sqrt(param_size) ~ 0.5 % error for a param size of 50 million (neglectable)
    hidden_size1 = math.floor(param_size ** (1/2))
    hidden_size2 = hidden_size1  
    
    # Create models
    model_pytorch = create_pytorch_model(input_size, hidden_size1, hidden_size2, output_size)
    opt_pytorch = get_pytorch_optimizer(model_pytorch, optimizer_name)

    # Generate data
    X, y = generate_data(n_samples=data_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data for PyTorch
    X_torch = torch.FloatTensor(X).to(device)
    y_torch = torch.FloatTensor(y).to(device)
    criterion = torch.nn.MSELoss()
    
    # Time PyTorch training
    start = time.time()
    for epoch in range(epochs):
        for i in range(data_size):
            outputs = model_pytorch(X_torch[i])
            loss = criterion(outputs, y_torch[i])
            opt_pytorch.zero_grad()
            loss.backward()
            opt_pytorch.step()
    pytorch_time = time.time() - start

    del model_pytorch
    del opt_pytorch
    del X, y
    gc.collect()

    # Regenerate data (avoid caching)
    X, y = generate_data(n_samples=data_size)

    model_torchLt = create_torchlt_model(input_size, hidden_size1, hidden_size2, output_size)
    opt_torchLt = get_torchlt_optimizer(model_torchLt, optimizer_name)
    
    # Prepare data for torchLt
    X_tensors = [torchLt.PyTensor(values=x.reshape(1, 1), shape=(1, 1), _track_gradient=True) for x in X]
    y_tensors = [torchLt.PyTensor(values=y_i.reshape(1, 1), shape=(1, 1), _track_gradient=True) for y_i in y]
    
    # Time torchLt training
    start2 = time.time()
    for epoch in range(epochs):
        for i in range(data_size):
            y_pred = model_torchLt(X_tensors[i])
            loss = y_pred.l2(y_tensors[i])
            loss.backward()
            opt_torchLt.asyncstep()
    torchLt_time = time.time() - start2

    del model_torchLt
    del opt_torchLt
    del X, y
    gc.collect()

    return pytorch_time, torchLt_time


def plot_results(results: Dict[str, Dict[int, List[float]]]):
    # Prepare data in format suitable for Altair
    data = []
    for optimizer in results.keys():
        for size, (pytorch_time, torchLt_time) in results[optimizer].items():
            data.extend([
                {
                    'Parameters (M)': size/1_000_000,
                    'Time (seconds)': pytorch_time,
                    'Framework': 'PyTorch',
                    'Optimizer': optimizer
                },
                {
                    'Parameters (M)': size/1_000_000,
                    'Time (seconds)': torchLt_time,
                    'Framework': 'torchLt',
                    'Optimizer': optimizer
                }
            ])
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create separate charts for each optimizer
    charts = []
    for optimizer in results.keys():
        optimizer_data = df[df['Optimizer'] == optimizer]
        
        chart = alt.Chart(optimizer_data).mark_line(point=True).encode(
            x=alt.X('Parameters (M):Q', title='Model Parameters (millions)'),
            y=alt.Y('Time (seconds):Q', scale=alt.Scale(type='log')),
            color='Framework:N',
            strokeDash='Framework:N',
            tooltip=['Parameters (M)', 'Time (seconds)', 'Framework']
        ).properties(
            width=400,
            height=300,
            title=f'{optimizer} Optimizer Performance'
        ).interactive()
        
        charts.append(chart)
    
    # Combine charts into a grid layout
    grid = alt.vconcat(*[alt.concat(*charts[i:i+2]) for i in range(0, len(charts), 2)])
    
    # Save the charts
    grid.save('benchmark_results.html')
    print("\nResults saved to 'benchmark_results.html'")


def main():
    param_sizes = range(25_000_000, 100_000_001, 25_000_000)
    #param_sizes = range(1_000_000, 2_000_001, 1_000_000)
    optimizers = ['Adam', 'RMSProp', 'Momentum', 'SGD']
    results = {opt: {} for opt in optimizers}
    
    for optimizer in optimizers:
        print(f"\nTesting {optimizer} optimizer...")
        for size in param_sizes:
            print(f"Testing model with {size/1_000_000:.1f}M parameters...")
            times = []
            for run in range(3):
                pytorch_time, torchLt_time = benchmark_training(size, optimizer)
                times.append((pytorch_time, torchLt_time))
            
            # Average the times
            avg_pytorch = sum(t[0] for t in times) / 3
            avg_torchLt = sum(t[1] for t in times) / 3
            
            results[optimizer][size] = (avg_pytorch, avg_torchLt)
            print(f"Average times - PyTorch: {avg_pytorch:.2f}s, torchLt: {avg_torchLt:.2f}s")
    
    plot_results(results)
    print("\nBenchmark complete. Results saved to 'benchmark_results.html'")


if __name__ == "__main__":
    main()