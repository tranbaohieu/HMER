import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from Discriminator import Discriminator


class Adversarial_Loss(nn.Module):
	def __init__(self, lambda_adv):
		super(Adversarial_Loss, self).__init__()
		self.lambda_adv = lambda_adv
		pass
	def forward(self, input_p, input_h):
		dis_p = input_p*torch.log(input_p)
		# print("input_p:", input_p)
		# print("input_h:", input_h)
		dis_h = torch.log(torch.ones_like(input_h) - input_h)
		adv_loss = dis_h + dis_p
		# print("adv_loss: ", adv_loss)
		return torch.sum(self.lambda_adv*adv_loss)
		# pass

class Loss_D(nn.Module):
	"""docstring for Loss_D"""
	def __init__(self):
		super(Loss_D, self).__init__()
	def forward(self, input_h):
		return - input_h*torch.log(input_h)
		pass
		