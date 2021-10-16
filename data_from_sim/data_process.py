import pickle

def result_collect(path):
	i = 0
	data_days = {}
	data_index = {}
	for line in open(path):
		if i<2:
			i+=1
			continue
		time = line.split(' ')[0]
		temp = line.split(' ')[1]
		data = temp.split('\t')
		data = data[2:]
		data = list(map(lambda x:float(x),data))
		new_data = [data[i*5+4] for i in range(5)]


		length_data_index = len(data_index)
		data_index[length_data_index] = time

		data_days[time] = {}
		data_days[time]['data'] = new_data


		if not i//100:
			print('every body go')


	with open('data_days','wb') as pkl:
		pickle.dump(data_days,pkl)
	with open('data_index', 'wb') as pkl2:
		pickle.dump(data_index, pkl2)

def data_filter(path):
	"""
	select the adults data from the final_data_new file
	:param path:
	:return:
	"""
	with open(path,'rb') as pkl:
		final_data_new = pickle.load(pkl)
	for k,v in final_data_new.items():
		v['start'] = v['start'][-1]
		v['target'] = v['target'][-1]

	with open('data_adults','wb') as pkl2:
		pickle.dump(final_data_new,pkl2)

if __name__ == "__main__":
	data_filter('final_data_new')