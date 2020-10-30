import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from Discriminator import Discriminator


class Adversarial_Loss(nn.Module):
	def __init__(self, lambda_adv):
		self.loss_adv = nn.NLLLoss()
		self.lambda_adv = lambda_adv
		pass
	def forward(self, loss_ch, loss_cp, input, target):
		adv_loss = loss_adv(input, target)
		return loss_ch + loss_cp + lambda_adv*adv_loss
		pass