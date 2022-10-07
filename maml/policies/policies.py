import math

import torch
from torch import nn
from torch.distributions import Normal, Categorical

EPSILON = 1e-6

def weight_init(layer):
	if isinstance(layer, nn.Linear):
		nn.init.xavier_uniform_(layer.weight)
		layer.bias.data.zero_()

	return layer

class GaussianPolicy(nn.Module):

	def __init__(self, in_size, out_size, hidden_units=None, activation='relu', device='cpu'):

		super(GaussianPolicy, self).__init__()
		self.device = device
		if hidden_units is None:
			self.hidden_units = [100, 100]
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
		self.logstd  = nn.Parameter(torch.zeros(1, out_size))
		

	def sample(self, observations):

		dist = self(observations)
		return dist.sample()

	def logprobs(self, observations, actions):

		dist = self(observations)
		return dist.log_prob(actions)

	def forward(self, observations):

		mean = self.mean(observations)
		scale= torch.exp(torch.clamp(self.logstd, min=math.log(EPSILON)))
		scale = scale.expand_as(mean)
		return Normal(loc=mean, scale=scale)


class CategoricalPolicy(nn.Module):

	def __init__(self, in_size, out_size, hidden_units=None, activation='relu', device='cpu'):

		super(CategoricalPolicy, self).__init__()
		self.device = device
		if hidden_units is None:
			self.hidden_units = [100, 100]
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


	def sample(self, observations):
		dist = self(observations)
		return dist.sample()

	def logprobs(self, observations, actions):

		return self(observations).log_prob(actions)

	def forward(self, observations):

		return Categorical(logits = self.mean(observations))
