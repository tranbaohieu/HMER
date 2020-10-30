import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, in_dim, out_dim):
		super(Discriminator, self).__init__()
		self.Linear1 = nn.Linear(in_dim, out_dim)
		self.Relu = nn.ReLU()
		self.Linear2 = nn.Linear(out_dim, 1)
		self.Sigmoid = nn.Sigmoid()

	def forward(self, input):
		x = self.Linear1(input)
		x = self.Relu(x)
		x = self.Linear2(x)
		x = self.Sigmoid(x)
		return x


