"""
From UC Berkeley CS 285 course
"""
from typing import Union
from collections.abc import Iterable

import torch
from torch import nn
import numpy as np
import pandas as pd

from dataset.common import logger

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(beta=5),
    'identity': nn.Identity(),
    'gelu': nn.GELU()
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int | tuple,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        dropout=0.2,
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for i in range(n_layers):
        if isinstance(size, int):
            layers.append(nn.Linear(in_size, size))
            in_size = size
        elif isinstance(size, Iterable):
            layers.append(nn.Linear(in_size, size[i]))
            in_size = size[i]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(activation)
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu and gpu_id is not None:
        device = torch.device("cuda:" + str(gpu_id))
        logger.info("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        logger.info("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs) -> torch.Tensor:
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor) -> np.ndarray:
    return tensor.to('cpu').detach().numpy()


class X_y_dataset(torch.utils.data.Dataset):
    def __init__(self, *args) -> None:
        self.tensors = []
        super().__init__()
        if isinstance(args[0], torch.Tensor):
            self.tensors = args
        elif isinstance(args[0], np.ndarray):
            for arg in args:
                self.tensors.append(from_numpy(arg))
        # elif isinstance(X, pd.DataFrame):
        #     self.X = from_numpy(X.to_numpy())
        #     self.y = from_numpy(y.to_numpy())
        #     if A:
        #         self.A = from_numpy(A.to_numpy())
    
    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)
