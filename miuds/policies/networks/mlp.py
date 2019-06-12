from collections import OrderedDict
import torch
import torch.nn as nn


class MLP(nn.Module):
    '''
    Multi Layer Perceptron
    '''
    def __init__(self, input_size, output_size, hidden_size,
                 num_hidden_layers=0, activation='ReLU'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.activation = getattr(nn, activation)()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        if self.num_hidden_layers > 0:
            raise NotImplementedError('Not support num_hidden_layers > 0 now.')

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        if self.num_hidden_layers > 0:
            # TODO
            pass
        x = self.output_layer(x)
        return x
