import torch
import torch.nn as nn
import torch.nn.functional as F 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionDecoder(nn.Module):
	def __init__(self, hidden_size, output_size, encoder_output):
		super(AttentionDecoder, self).__init__()
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(self.output_size, 256)
		
		pass 

	def forward(self):
		pass

	def initHidden(self):
		pass 

