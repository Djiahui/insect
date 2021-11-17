import pickle

def write_in_csv():
	i = 0
	with open('result.csv','w') as f:

		for line in open('Output_Results.txt'):
			if i<1:
				i+=1
				continue

			if i==1:
				temp = line.split('\t')
				i+=1
			else:
				temp = line.split(' ')
				temp = [temp[0]] + temp[1].split('\t')
			temp = ','.join(temp)
			f.write(temp)

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

def temp_data_collect():
	with open('final_data_new','rb') as pkl:
		data = pickle.load(pkl)

	temperature_data = {}
	count = 0
	for k,v in data.items():
		if not v['data'] in temperature_data and count==2:
			temperature_data[v['data']] = v['temp']
			count = 0
		elif v['data'] in temperature_data:
			continue
		else:
			count += 1

	with open('temperature_data','wb') as pkl2:
		pickle.dump(temperature_data,pkl2)
# def two_dimension_data_process():
# 	#温度只考虑十二点的温度的版本
# 	i = 0
#
# 	for line in open('Output_Results.txt'):
# 		if i<2:
# 			i+=1
# 			continue
# 		time = line.split(' ')[0]
# 		temp = line.split(' ')[1]
# 		data = temp.split('\t')
# 		data = data[2:]
def only_adult_1():
	#只要adult1的数据
	adults_1_only = {}
	with open('data_adults','rb') as pkl:
		data = pickle.load(pkl)

		n = len(data)

		for i in range(0,n,5):
			m = len(adults_1_only)
			adults_1_only[m] = {}
			adults_1_only[m] = data[i]
	with open('data_adults_only_1','wb') as pkl2:
		pickle.dump(adults_1_only,pkl2)
if __name__ == "__main__":
	only_adult_1()