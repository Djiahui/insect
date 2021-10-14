import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pickle

class pre_net(nn.Module):
	def __init__(self):
		super(pre_net, self).__init__()
		self.linear = nn.Linear(25,1)
	def forward(self,input):
		out = self.linear(input)

		return out

class data_set(Dataset):
	def __init__(self):
		with open('data') as pkl:
			pass

	def __len__(self):
		pass

	def __getitem__(self, index):
		pass