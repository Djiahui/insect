import pickle
import matplotlib.pyplot as plt
import numpy as np
with open('data.pkl','rb') as pkl:
	data = pickle.load(pkl)

new_data = []
for k,v in data.items():
	sum_k = 0
	for l in v['sample']:
		sum_k += sum(l)

	new_data.append([sum_k,v['label']])


temp = np.array(new_data)

plt.scatter(temp[:,0],temp[:,1])

plt.xlabel('the number of traps')
plt.ylabel('the loss of food')
plt.show()

exit()