import pickle
from tqdm import tqdm
with open('surrogate_model/data_sample.pkl','rb') as pkl:
	temp = pickle.load(pkl)

with open('surrogate_model/data.csv','w') as f:
	temp_s = 'order'
	temp_days = [str(i+1) for i in range(len(temp[0]['label']))]
	temp_c = 'capture'

	temp_name = [temp_s] + temp_days + [temp_c]

	temp_name = ','.join(temp_name)+'\n'

	f.write(temp_name)

	for i in tqdm(range(len(temp))):
		temp_k = str(i)
		v = temp[i]
		temp_in_machine = list(map(lambda x:str(x),v['label']))
		temp_in_trap = str(v['captured'])

		temp_words = [temp_k] + temp_in_machine + [temp_in_trap]
		temp_words = ','.join(temp_words) + '\n'
		f.write(temp_words)

		temp_words_2 = [' '] + list(map(lambda x:str(x),v['insect_nums'])) + [' ']
		temp_words_2 = ','.join(temp_words_2) + '\n'
		f.write(temp_words_2)







