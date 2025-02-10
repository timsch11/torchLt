import sys
import os


# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from Tensor import *

import time
import random


DATASIZE = 100
EPOCHS = 1

NEURONS_I = 1
NEURONS_H1 = 100000
NEURONS_H2 = 1000
NEURONS_O = 1

"""Data to train our neural network: Input is in [0, 100] and corresponding label in [0, 1000]: both are totally random"""
"""IMPORTANT: This should only show the speed of the network"""


inp = [PyTensor(values=[random.random() * 100], shape=(NEURONS_I, 1), _track_gradient=True) for i in range(DATASIZE)]

w0 = PyTensor(shape=(NEURONS_H1, NEURONS_I), _track_gradient=True, kaimingHeInit=True)
b0 = PyTensor([[0 for i in range(NEURONS_H1)]], (NEURONS_H1, 1), _track_gradient=True)

w1 = PyTensor(shape=(NEURONS_H2, NEURONS_H1), _track_gradient=True, kaimingHeInit=True)
b1 = PyTensor([[0 for i in range(NEURONS_H2)]], (NEURONS_H2, 1), _track_gradient=True)

w2 = PyTensor(shape=(NEURONS_O, NEURONS_H2), _track_gradient=True, kaimingHeInit=True)
b2 = PyTensor([[0 for i in range(NEURONS_O)]], (NEURONS_O, 1), _track_gradient=True)

labels = [PyTensor(values=[random.random() * 1000], shape=(1, 1), _track_gradient=True) for i in range(DATASIZE)]


def train(epochs: int):
    for _ in range(epochs):
        for i in range(DATASIZE):
            # input layer
            a0 = ((w0 @ inp[i]) + b0).relu()

            # hidden layer
            a1 = ((w1 @ a0) + b1).relu()

            # output layer
            a2 = ((w2 @ a1) + b2)

            # calculate loss
            loss = a2.l2(labels[i])

            # calculate gradients through backpropagation
            loss.backward()

            # apply gradient descent
            w0.sgd(lr=0.001)
            b0.sgd(lr=0.001)
            w1.sgd(lr=0.001)
            b1.sgd(lr=0.001)
            w2.sgd(lr=0.001)
            b2.sgd(lr=0.001)


def evaluate(_inp: PyTensor, label: PyTensor) -> tuple[PyTensor, PyTensor]:
    """returns a tuple of (prediction, loss)"""
    # input layer
    a0 = ((w0 @ _inp) + b0).relu()

    # hidden layer
    a1 = ((w1 @ a0) + b1).relu()

    # output layer
    a2 = ((w2 @ a1) + b2)

    # calculate loss
    loss = a2.l2(label)

    return a2, loss


if __name__ == '__main__':
    print("### Training ###")
    paramters = NEURONS_I * NEURONS_H1 + NEURONS_H1 * NEURONS_H2 + NEURONS_H2 * NEURONS_O + NEURONS_H1 + NEURONS_H2 + NEURONS_O
    print(f"Model parameters: {paramters / 1000000} million")
    t1 = time.time()
    train(EPOCHS)
    print(f"Iterations per second: {(DATASIZE * EPOCHS) / (time.time() - t1)}\n")