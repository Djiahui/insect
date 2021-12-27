import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pickle
import matplotlib.pyplot as plt
import os
torch.manual_seed(1234)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class pre_net(nn.Module):
	def __init__(self):
		super(pre_net, self).__init__()
		self.l = nn.Sequential(nn.Linear(2,16),
							   nn.ReLU(),
							   nn.Linear(16,1))
		# self.seq = nn.Sequential(nn.Linear(25,64),
		# 						 nn.ReLU(),
		# 						 nn.Linear(64,32),
		# 						 nn.ReLU(),
		# 						 nn.Linear(32,16),
		# 						 nn.ReLU(),
		# 						 nn.Linear(16,8),
		# 						 nn.ReLU(),
		# 						 nn.Linear(8,1))
	def forward(self,input):


		return self.l(input)

class data_set(Dataset):
	def __init__(self):
		with open('data_adults','rb') as pkl:
			self.data = pickle.load(pkl)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		temp = self.data[index]

		start = torch.tensor(temp['start'])
		wendu = torch.tensor(temp['temp']).mean()
		target = torch.tensor(temp['target'])

		return {'start':start,'temp':wendu,'target':target}
def train(net,train_loader,loss_fun,optim):
	total_loss = []
	for _,data in enumerate(train_loader):
		s = data['start']
		temp = data['temp']
		target = data['target'].view(-1, 1)

		sample = torch.cat((s.view(-1, 1), temp.view(-1,1)), 1)
		predict_number = net(sample)
		loss = loss_fun(predict_number, target)


		optim.zero_grad()
		loss.backward()
		optim.step()

def eval(net,test_dataloader,loss_fun):
	for test_data in test_dataloader:
		s_test = test_data['start']
		temp_test = test_data['temp']
		target_test = test_data['target'].view(-1, 1)

		sample_test = torch.cat((s_test.view(-1, 1), temp_test.view(-1,1)), 1)
		predict_number_test = torch.clamp(net(sample_test),1,1e6)
		loss_test = torch.sqrt(loss_fun(torch.log(predict_number_test), torch.log(target_test))).item()

	return loss_test


def main():
	dataset = data_set()
	# dataloader = DataLoader(dataset,256,shuffle=True)
	n = len(dataset)
	train_num = int(n * 0.85)
	test_num = n - train_num
	print(train_num,test_num)
	train_data, test_data = torch.utils.data.random_split(dataset, [train_num, test_num])

	dataloader_train = DataLoader(train_data,256,shuffle=True)
	dataloader_test = DataLoader(train_data,test_num,shuffle=True)

	net = pre_net()
	loss_fun = torch.nn.MSELoss()

	optim = torch.optim.Adam(net.parameters(),0.001)
	test_ls = []

	for ep in range(100):
		train(net,dataloader_train,loss_fun,optim)
		test_loss = eval(net,dataloader_test,loss_fun)
		print(test_loss)
		test_ls.append(test_loss)

	plt.plot(range(len(test_ls)),test_ls)
	plt.xlabel('epochs')
	plt.ylabel('error')
	# plt.savefig('../png/error.png')
	plt.show()

	exit(0)

	torch.save(net.state_dict(),'regression_model_parameters_non_linear.pkl')

if __name__ == "__main__":
	main()