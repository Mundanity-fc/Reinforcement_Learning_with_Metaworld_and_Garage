import torch
from torch import nn

def weight_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.zero_()
    return layer

class ValueFun(nn.Module):

    def __init__(self, in_size, out_size, hidden_units=None, activation='relu', device='cpu'):

        super(ValueFun, self).__init__()
        self.device = device
        if hidden_units is None:
            self.hidden_units = [32, 32]
        else:
            self.hidden_units = hidden_units

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        layers = [weight_init(nn.Linear(in_size, self.hidden_units[0])), self.activation]

        for i_size, o_size in zip(self.hidden_units[:-1], self.hidden_units[1:]):
            layers.append(weight_init(nn.Linear(i_size, o_size)))
            layers.append(self.activation)

        layers.append(weight_init(nn.Linear(o_size, out_size)))

        self.mean = nn.Sequential(*layers)

        self.loss = nn.MSELoss()
        

    def forward(self, observations):

        values = self.mean(observations)
        return values
        
    def value_loss(self, observations, returns):

        values = self.mean(observations).squeeze(2)
        loss = self.loss(values, returns)

        return loss