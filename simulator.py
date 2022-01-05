import random

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
def draw_new(alive,involved,intrap,traps):
	alivepos = np.vstack([[x[1],200-x[0]] for x in alive])
	involvedpos = np.vstack([[x[1], 200 - x[0]] for x in involved])
	intrappos = np.vstack([[x[1], 200 - x[0]] for x in intrap])
	trap_pos = np.vstack(list(map(lambda x: [x.pos[1], 200 - x.pos[0]], traps)))

	plt.scatter(alivepos[:,0],alivepos[:,1],c = 'g')
	plt.scatter(involvedpos[:, 0], involvedpos[:, 1], c='b')
	plt.scatter(intrappos[:, 0], intrappos[:, 1], c='purple')
	plt.scatter(trap_pos[:, 0], trap_pos[:, 1], c='r', alpha=0.5)

	plt.grid()
	plt.xticks(np.arange(0, 200, 10))
	plt.yticks(np.arange(0, 200, 10))
	plt.show()




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

	with open('data_from_sim/temperature_data', 'rb') as pkl:
		temp_data = pickle.load(pkl)

	day_count = 0
	in_machine_nums = [0 for _ in range(iteration)]
	insect_nums = [0 for _ in range(iteration)]

	for _, temp in temp_data.items():

		insect_position_alive,insect_position_involved,insect_position_traps = pops.update(traps, temp)

		# draw_new(insect_position_alive,insect_position_involved,insect_position_traps,traps)

		in_machine_nums[day_count] = pops.env.in_machine_num
		insect_nums[day_count] = len(pops.populations)
		pops.env.update()

		if draw_or_not:
			draw(pops, traps,day_count)
			exit()

		day_count += 1
		if day_count == iteration:
			break

	return in_machine_nums,pops.env.in_trap_num,insect_nums


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
		insects_in_machine,insects_in_trap,insect_nums = simulate(matrix, insect_iteration, new_insect_pops)
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
	x, y = 100, 100
	step = 10

	env = entity.screen(x, y, step)
	pops = entity.insect_population(pop_num, env)

	# simulate(pop_num,matrix,iteration)
