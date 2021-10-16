import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pickle

torch.manual_seed(1234)

class pre_net(nn.Module):
	def __init__(self):
		super(pre_net, self).__init__()
		self.linear = nn.Linear(25,1)
	def forward(self,input):
		out = self.linear(input)

		return out

class data_set(Dataset):
	def __init__(self):
		with open('data_adults','rb') as pkl:
			self.data = pickle.load(pkl)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		temp = self.data[index]

		start = torch.tensor(temp['start'])
		wendu = torch.tensor(temp['temp'])
		target = torch.tensor(temp['target'])

		return {'start':start,'temp':wendu,'target':target}

def main():
	dataset = data_set()
	dataloader = DataLoader(dataset,64,shuffle=True)
	# n = len(dataset)
	# train_num = int(n * 0.85)
	# test_num = n - train_num
	# train_data, test_data = torch.utils.data.random_split(dataset, [train_num, test_num])
	#
	# dataloader_train = DataLoader(train_data,64,shuffle=True)
	# dataloader_test = DataLoader(train_data,64,shuffle=True)

	net = pre_net()
	loss_fun = torch.nn.SmoothL1Loss()

	optim = torch.optim.Adam(net.parameters(),0.1)

	for batch_th in range(100):
		for ith,data in enumerate(dataloader):
			s = data['start']
			temp = data['temp']
			target = data['target'].view(-1,1)

			sample = torch.cat((s.view(-1,1),temp),1)
			predict_number = net(sample)
			loss = loss_fun(predict_number,target)

			optim.zero_grad()
			loss.backward()
			optim.step()

			if not ith//10:
				local_loss = []
				for _, data  in enumerate(dataloader):
					s_test = data['start']
					temp_test = data['temp']
					target_test = data['target'].view(-1,1)

					sample_test = torch.cat((s_test.view(-1,1), temp_test), 1)
					predict_number_test = net(sample_test)
					loss_test = loss_fun(predict_number_test, target_test).item()

					local_loss.append(loss_test)
				print(torch.tensor(local_loss).mean().item()/dataset.__len__())











if __name__ == "__main__":
	main()