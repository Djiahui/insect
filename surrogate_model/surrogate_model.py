import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
import pickle
import matplotlib.pyplot as plt

class dataset(Dataset):
	def __init__(self):
		with open('../data2.pkl', 'rb') as pkl:
			self.total_data = pickle.load(pkl)

	def __len__(self):
		return len(self.total_data)
	def __getitem__(self, index):
		sample = torch.tensor(self.total_data[index]['sample']).float()
		label = torch.tensor(self.total_data[index]['label']).float()

		return {'sample':sample,'label':label}



class surrogate_net(nn.Module):
	def __init__(self):
		super(surrogate_net, self).__init__()
		self.con0 = nn.Conv2d(1,1,4,2)
		self.con1 = nn.Conv2d(1,1,kernel_size=2,stride=2)
		self.con2 = nn.Conv2d(1,1,kernel_size=2,stride=2)
		self.con3 = nn.Conv2d(1,1,kernel_size=2,stride=1)
		self.relu = nn.ReLU()

	def forward(self,input):

		input = input.unsqueeze(1)
		out = self.con0(input)
		out = self.relu(out)
		out = self.con1(out)
		out = self.relu(out)
		out = self.con2(out)
		out = self.relu(out)
		out = self.con3(out)
		out = self.relu(out)

		return out.squeeze(1).squeeze(1).squeeze(1)
def main():
	data = dataset()
	train_size = int(0.95*len(data))
	test_size = len(data)-train_size
	train_data,test_data = random_split(data,[train_size,test_size])

	train_loader = DataLoader(train_data,batch_size=16)
	test_loader = DataLoader(test_data,batch_size=16)

	mynet = surrogate_net()
	optimizer = torch.optim.Adam(mynet.parameters(),lr=0.01)
	loss_fun = nn.MSELoss()
	total_loss = []
	for e in range(2):
		for i,batch in enumerate(train_loader):
			sample = batch['sample']
			label = batch['label']

			pre = mynet(sample)
			loss = loss_fun(pre,label)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i%5==0:
				temp_loss = []
				for t,t_batch in enumerate(test_loader):
					t_sample = t_batch['sample']
					t_label = t_batch['label']
					t_pre = mynet(t_sample)
					t_loss = loss_fun(t_pre,t_label)

					temp_loss.append(t_loss.item())

				total_loss.append(sum(temp_loss)/len(temp_loss))





	plt.plot(range(len(total_loss)),total_loss)
	plt.xlabel('the number of evaluation')
	plt.ylabel('the average loss')
	plt.show()
	exit()




if __name__ =="__main__":
	# main()
	# exit()
	input = torch.rand(1,1,21,21)

	con = nn.Conv2d(1,1,4,2)
	con1 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
	con2 = nn.Conv2d(1, 1, kernel_size=2, stride=2)
	con3 = nn.Conv2d(1, 1, kernel_size=2, stride=1)

	out = con(input)
	out = con1(out)
	out = con2(out)
	out =con3(out)

	exit(0)

