import random

import numpy
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import entity
import pickle
import os
from parameters import Parameters

def draw(pops, traps,day_count):
	"""
	# the Coordinate transformation is applied such as [0,10]--[10,200]
	:param pops:
	:param traps:
	:return:
	"""
	poses = list(map(lambda x: [x.pos[1], 200 - x.pos[0]], pops.populations))
	poses = np.vstack(poses)

	trap_pos = list(map(lambda x: [x.pos[1], 200 - x.pos[0]], traps))
	trap_pos = np.vstack(trap_pos)
	plt.scatter(poses[:, 0], poses[:, 1])
	plt.scatter(trap_pos[:, 0], trap_pos[:, 1], c='r', alpha=0.5)
	# plt.scatter(trap_pos[:,0],trap_pos[:,1],s=2000,c='r',alpha=0.5)
	plt.title(str(day_count))
	plt.grid()
	plt.xticks(np.arange(0, 200, 10))
	plt.yticks(np.arange(0, 200, 10))
	plt.show()
def draw_new(alive,involved,intrap,traps,times,ith,day_count):
	if alive:
		alivepos = np.vstack([[x[1],200-x[0]] for x in alive])
		plt.scatter(alivepos[:, 0], alivepos[:, 1], c='g',marker='*')
	if involved:
		involvedpos = np.vstack([[x[1], 200 - x[0]] for x in involved])
		plt.scatter(involvedpos[:, 0], involvedpos[:, 1], c='b',marker='x')
	if intrap:
		intrappos = np.vstack([[x[1], 200 - x[0]] for x in intrap])
		plt.scatter(intrappos[:, 0], intrappos[:, 1], c='purple',marker='+')
	trap_pos = np.vstack(list(map(lambda x: [x.pos[1], 200 - x.pos[0]], traps)))
	plt.scatter(trap_pos[:, 0], trap_pos[:, 1], c='r', alpha=0.5,s = 100)
	try:
		plt.grid()
		plt.xticks(np.arange(0, 210, 10))
		plt.yticks(np.arange(0, 210, 10))
		name = 'png/temp/'+ str(ith) + '-' +str(times) +'_' +str(day_count) +'.png'
		plt.savefig(name)
		plt.show()
	except:
		print(str(ith)+'th'+str(day_count)+'days'+'problem')




def trap_generate(matrix, x, y, step):
	x_num = x // step + 1
	y_num = y // step + 1
	traps = []
	for i in range(x_num):
		for j in range(y_num):
			if matrix[i,j]:
				traps.append(entity.trap([i * step, j * step]))

	return traps


def simulate(matrix, iteration, pops,draw_or_not = False):
	traps = trap_generate(matrix, pops.env.x, pops.env.y, pops.env.step)
	pops.generate(traps)
	if draw_or_not:
		draw(pops, traps,0)

	with open('data_from_sim/new_temp_data', 'rb') as pkl:
		temp_data = pickle.load(pkl)

	day_count = 0
	in_machine_nums = [0 for _ in range(iteration)]
	insect_nums = [0 for _ in range(iteration)]
	new_gene = [0 for _ in range(iteration)]
	in_traps_num = [0 for _ in range(iteration)]

	days = []
	temps = []
	for day,temp in temp_data.items():
		days.append(day)
		temps.append(temp)

	num_days = len(days)
	season = Parameters.season
	if not season:
		while True:
			random_start = random.randint(0, num_days - 1)
			day = days[random_start]
			if random_start+iteration<num_days:
				break
	elif season=='s':
		while True:
			random_start = random.randint(0, num_days - 1)
			day = days[random_start]
			day_split = day.split('/')
			if day_split[1] in ['1','2','3','4'] and random_start+iteration<num_days:
				break
	else:
		while True:
			random_start = random.randint(0, num_days - 1)
			day = days[random_start]
			day_split = day.split('/')
			if day_split[1] in ['7', '8', '9', '10'] and day_split[0] != '2021'and random_start+iteration<num_days:
				break


	flag = False
	for itera in range(random_start,random_start+iteration):
		_,_,insect_position_traps = pops.update(traps,temps[itera])
		in_machine_nums[itera-random_start] = pops.env.in_machine_num
		insect_nums[itera-random_start] = len(pops.populations)
		in_traps_num[itera-random_start] = len(insect_position_traps) if insect_position_traps else 0
		new_gene[itera-random_start] = pops.new
		pops.env.update()

		if Parameters.Treatment:
			if not ((itera-random_start)+1)%10:

				temp_in_machine =  in_machine_nums[itera-random_start]
				temp_probaility = 0 if not temp_in_machine else 1 / (2 * (1 + np.exp(-temp_in_machine)))
				if not temp_probaility:
					cost = 0
				elif temp_probaility>Parameters.threshold:
					cost = (1+Parameters.discount_q)*temp_probaility
				else:
					cost = Parameters.discount_p*temp_probaility
				loss = cost*8064
				if loss>1158:
					flag = True
					print(str(itera-random_start)+'enough'+'kill')
					break

				# probaility = [0 if not x else 1 / (2 * (1 + np.exp(-x))) for x in temp_in_machine]
				# cost = [(1 + Parameters.discount_q)*x if x > Parameters.threshold else (Parameters.discount_p*x) for x in
				# 		probaility]
				#
				# for index, pro in enumerate(probaility):
				# 	if not pro:
				# 		cost[index] = 0
				#
				# temp_loss = sum(cost)*8064
				# print(temp_loss)
				# norm
				# temp_loss /= (1 + Parameters.discount_q) * Parameters.insect_iteration

				# if temp_loss>1158:
				# 	flag = True
				# 	print('kill!!!!!!!!!!!!!!!!!!!!!!!')
				# 	break

	return in_machine_nums, in_traps_num, insect_nums, new_gene,flag

	# for _, temp in temp_data.items():
	# 	pops.update(traps, temp)
	#
	# 	in_machine_nums[day_count] = pops.env.in_machine_num
	# 	insect_nums[day_count] = len(pops.populations)
	# 	new_gene[day_count] = pops.new
	# 	pops.env.update()
	#
	# 	if draw_or_not:
	# 		draw(pops, traps,day_count)
	# 		exit()
	#
	# 	day_count += 1
	# 	if day_count == iteration:
	# 		break
	#
	# return in_machine_nums,pops.env.in_trap_num,insect_nums,new_gene

def picture_simulate(matrix, iteration, pops,times,ith):
	traps = trap_generate(matrix, pops.env.x, pops.env.y, pops.env.step)
	pops.generate(traps)
	with open('data_from_sim/new_temp_data', 'rb') as pkl:
		temp_data = pickle.load(pkl)

	day_count = 0
	in_machine_nums = [0 for _ in range(iteration)]
	insect_nums = [0 for _ in range(iteration)]
	new_gene = [0 for _ in range(iteration)]

	days = []
	temps = []
	for day,temp in temp_data.items():
		days.append(day)
		temps.append(temp)

	num_days = len(days)
	season = Parameters.season
	if not season:
		while True:
			random_start = random.randint(0, num_days - 1)
			day = days[random_start]
			if random_start+iteration<num_days:
				break
	elif season=='s':
		while True:
			random_start = random.randint(0, num_days - 1)
			day = days[random_start]
			day_split = day.split('/')
			if day_split[1] in ['1','2','3','4'] and random_start+iteration<num_days:
				break
	else:
		while True:
			random_start = random.randint(0, num_days - 1)
			day = days[random_start]
			day_split = day.split('/')
			if day_split[1] in ['7', '8', '9', '10'] and day_split[0] != '2021'and random_start+iteration<num_days:
				break

	for itera in range(random_start,random_start+iteration):
		insect_position_alive,insect_position_involved,insect_position_traps = pops.update(traps,temps[itera])
		draw_new(insect_position_alive, insect_position_involved, insect_position_traps, traps, times, ith, itera-random_start)
		in_machine_nums[itera-random_start] = pops.env.in_machine_num
		insect_nums[itera-random_start] = len(pops.populations)
		pops.env.update()

		if Parameters.Treatment:
			if Parameters.Treatment:
				if not ((itera - random_start) + 1) % 10:

					temp_in_machine = in_machine_nums[itera-random_start]
					temp_probaility = 0 if not temp_in_machine else 1 / (2 * (1 + np.exp(-temp_in_machine)))
					if not temp_probaility:
						cost = 0
					elif temp_probaility > Parameters.threshold:
						cost = (1 + Parameters.discount_q) * temp_probaility
					else:
						cost = Parameters.discount_p * temp_probaility
					loss = cost * 8064
					if loss > 1158:
						print(str(itera - random_start) + 'enough'+ 'kill')
						flag = True
						break

	return in_machine_nums, pops.env.in_trap_num, insect_nums

	# traps = trap_generate(matrix, pops.env.x, pops.env.y, pops.env.step)
	# pops.generate(traps)
	# with open('data_from_sim/new_temp_data', 'rb') as pkl:
	# 	temp_data = pickle.load(pkl)
	#
	# day_count = 0
	# in_machine_nums = [0 for _ in range(iteration)]
	# insect_nums = [0 for _ in range(iteration)]
	#
	# for _, temp in temp_data.items():
	# 	insect_position_alive,insect_position_involved,insect_position_traps = pops.update(traps, temp)
	# 	draw_new(insect_position_alive,insect_position_involved,insect_position_traps,traps,times,ith,day_count)
	#
	# 	in_machine_nums[day_count] = pops.env.in_machine_num
	# 	insect_nums[day_count] = len(pops.populations)
	# 	pops.env.update()
	#
	# 	day_count += 1
	# 	if day_count == iteration:
	# 		break
	#
	#
	# return in_machine_nums,pops.env.in_trap_num,insect_nums


def matrix_generate(x, y, step, test=True):
	if test:
		with open('matrix_test.pkl','rb') as pkl:
			matrix = pickle.load(pkl)
		return np.array(matrix)


	x_num = x // step + 1
	y_num = y // step + 1

	matrix = [[0] * y_num for _ in range(x_num)]
	temp = np.random.rand()
	if temp < 0.2:
		for i in range(x_num):
			for j in range(y_num):
				# only for current screen
				if i < 6 and j < 12:
					continue
				r1 = np.random.rand()
				if r1 < 0.01:
					matrix[i][j] = 1
	elif 0.2 < temp < 0.4:
		for i in range(x_num):
			for j in range(y_num):
				# only for current screen
				if i < 6 and j < 12:
					continue
				r1 = np.random.rand()
				if r1 < 0.01:
					matrix[i][j] = 1
	elif 0.4 < temp < 0.6:
		for i in range(x_num):
			for j in range(y_num):
				# only for current screen
				if i < 6 and j < 12:
					continue
				r1 = np.random.rand()
				if r1 < 0.2:
					matrix[i][j] = 1
	elif 0.6 < temp < 0.8:
		for i in range(x_num):
			for j in range(y_num):
				# only for current screen
				if i < 6 and j < 12:
					continue
				r1 = np.random.rand()
				if r1 < 0.25:
					matrix[i][j] = 1
	else:
		for i in range(x_num):
			for j in range(y_num):
				# only for current screen
				if i < 6 and j < 12:
					continue
				r1 = np.random.rand()
				r2 = np.random.rand()
				if r2 < r1:
					matrix[i][j] = 1
	return np.array(matrix)


def prob_cal(insect_in_machine):
	pass


def sample_generate(x, y, step, insect_num, sample_num, insect_iteration):
	env = entity.screen(x, y, step)
	# pops = entity.insect_population(insect_num, copy.deepcopy(env))

	# for _ in range(sample_num):
	for _ in tqdm(range(sample_num)):
		if not os.path.exists('surrogate_model\data_sample.pkl'):
			data = {}
			with open('surrogate_model\data_sample.pkl', 'wb') as pkl:
				pickle.dump(data, pkl)
		with open('surrogate_model\data_sample.pkl', 'rb') as pkl1:
			data = pickle.load(pkl1)

		pops = entity.insect_population(Parameters().get_random_insect_number(), copy.deepcopy(env))
		matrix = matrix_generate(env.x, env.y, env.step, Parameters.test)
		# no traps
		# matrix = [[0 for _ in range(21)] for _ in range(21)]
		# one matrix for testing

		new_insect_pops = copy.deepcopy(pops)
		insects_in_machine,insects_in_trap,insect_nums,new_nums,flag = simulate(matrix, insect_iteration, new_insect_pops)
		probaility = prob_cal(insects_in_machine)

		n = len(data)
		data[n] = {}
		data[n]['sample'] = matrix
		data[n]['label'] = insects_in_machine
		data[n]['captured'] = insects_in_trap
		data[n]['insect_nums'] = insect_nums

		with open('surrogate_model/data_sample.pkl', 'wb') as pkl:
			pickle.dump(data, pkl)




if __name__ == '__main__':
	pop_num = 100
	trap_num = 5

	iteration = 20
	x, y = 200, 200
	step = 10

	env = entity.screen(x, y, step)
	pops = entity.insect_population(pop_num, env)
	matrix = numpy.array([[0 for _ in range(21)] for _ in range(21)])
	simulate(matrix,20,pops)
