import matplotlib.pyplot as plt
import numpy
import numpy as np
import copy
import random

import torch
from data_from_sim.predict_net import pre_net
from surrogate_model.surrogate_model import surrogate_net
from parameters import Parameters
import simulator
from multiprocessing import Pool

class insect(object):
	def __init__(self, x, y, pos=None, direction=None, rate=None):
		if pos:
			self.pos = pos
		else:
			self.pos = np.random.rand(2) * 200

		if direction:
			self.direction = direction
		else:
			self.direction = np.random.rand(2)
		# self.direction = self.direction / np.linalg.norm(self.direction, ord=2)

		if rate:
			self.rate = rate
		else:
			self.rate = 4

		self.age = 0

		self.best_pos = np.zeros(2)
		self.best_fit = None
		self.fitness = None
		self.status = True
		self.stand_in_same_place = 0

		self.eat_num = 0.05

		self.involved = False
		self.in_trap = False

	def update(self, global_best, env, traps):

		self.age += 1
		if self.age>=14:
			self.status = False
			return

		temp1 = self.best_pos - self.pos
		temp2 = global_best.pos - self.pos

		pre_pos = [int(self.pos[0] // env.step), int(self.pos[1] // env.step)]
		self.direction = self.direction + random.random() * temp1 + random.random() * temp2

		self.pos = self.pos + np.linalg.norm(self.direction, ord=2)

		# self.direction = np.linalg.norm(temp1 + temp2, ord=2)+self.direction
		# if self.direction.sum():
		# 	self.pos = self.pos + self.direction * self.rate
		# 	self.direction = self.direction / np.linalg.norm(self.direction, ord=2)

		# if random.random()<0.01:
		# 	print('direction')
		# 	print(self.direction)

		# for current screen
		if 0 < self.pos[0] < 60 and 0 < self.pos[1] < 120:
			if pre_pos[0] < 6:
				self.pos[1] = 120
			else:
				self.pos[0] = 60

		self.pos[0] = min(env.x, self.pos[0])
		self.pos[0] = max(self.pos[0], 0)
		self.pos[1] = min(env.y, self.pos[1])
		self.pos[1] = max(0, self.pos[1])

		# if self.pos[0] < 0:
		# 	self.pos[0] = -self.pos[0]
		# if self.pos[0] > env.x:
		# 	self.pos[0] = 2 * env.x - self.pos[0]
		#
		# if self.pos[1] < 0:
		# 	self.pos[1] = -self.pos[1]
		# if self.pos[1] > env.y:
		# 	self.pos[1] = 2 * env.y - self.pos[1]
		self.living_test(traps)
		if not self.status:
			self.in_trap = True
			return

		cur_pos = [int(self.pos[0] // env.step), int(self.pos[1] // env.step)]
		self.stand_in_same_place += (1 if pre_pos == cur_pos else 0)
		repair_pos = [min(x, 19) for x in cur_pos]

		self.in_machine(repair_pos, env)
		if not self.status:
			self.involved = True
			return

		self.fitness = env.eva(self.pos, self.eat_num)
		if self.fitness > self.best_fit:
			self.best_fit = self.fitness
			self.best_pos = self.pos

	def living_test(self, traps):
		for trap in traps:
			dis = np.linalg.norm(self.pos - trap.pos, ord=2)

			if dis < trap.radius:
				self.status = False
				break

	def in_machine(self, pos, env):
		probability = env.capture_prob[pos[0], pos[1]]
		e_t = np.exp(self.stand_in_same_place/100)
		probability = (e_t + probability) / (1 + e_t + probability)
		probability = probability * Parameters.insect_fall_machine
		temp_random = np.random.random()
		if temp_random < probability:
			self.status = False
			env.in_machine_num += 1


class trap(object):
	def __init__(self, pos):
		self.pos = pos
		self.radius = 3


class Machine(object):
	def __init__(self, x, y, step):
		"""
		the parameter x and y are coordinate of block, thus the true coordinate of machine need to be calculated based on the step parameter.
		:param x:
		:param y:
		:param step:
		"""
		self.coordinate_x = x
		self.coordinate_y = y
		self.threshold = 5
		self.x = x * step + step // 2
		self.y = y * step + step // 2


class insect_population(object):
	def __init__(self, insect_num, env):
		self.insect_num = insect_num
		self.populations = []

		self.global_best = None
		self.env = env
		self.dead_num = 0

		self.regression_model = pre_net()
		self.regression_model.load_state_dict(torch.load('data_from_sim/regression_model_parameters_non_linear.pkl'))

	def generate(self, traps, num=None):
		if num == None:
			num = self.insect_num

		temp_num = 0
		while temp_num < num:
			temp = insect(self.env.x, self.env.y)
			if temp.pos[0] < 60 and temp.pos[1] < 120:
				continue
			temp.fitness = self.env.eva(temp.pos, temp.eat_num)
			temp.best_pos = temp.pos
			temp.best_fit = temp.fitness
			temp.living_test(traps)
			if not temp.status:
				self.env.in_trap_num += 1
				continue

			self.populations.append(temp)
			temp_num += 1

			if not self.global_best:
				self.global_best = temp
			else:
				if temp.fitness > self.global_best.fitness:
					self.global_best = copy.deepcopy(temp)

	def update(self, traps, temp):
		"""
		first generate new insects based on the current(yesterday) population
		second update the position
		third living test, machine test
		:param traps:
		:return:
		"""

		current_num = len(self.populations)

		input = torch.tensor([current_num] + [sum(temp) / len(temp)])
		# input = torch.tensor([current_num] + temp)
		temp_num = self.regression_model(input).item()
		predict_num = (int(temp_num)+1) if temp_num else 0
		# predict_num = int(self.regression_model(input).item())+1
		to_generate_num = max(0, predict_num - current_num)
		self.generate(traps, to_generate_num)

		if not self.populations:
			return None,None,None

		for temp in self.populations:
			temp.update(self.global_best, self.env, traps)

		insect_position_involved = [x.pos for x in self.populations if x.involved]
		insect_position_traps = [x.pos for x in self.populations if x.in_trap]

		self.populations = list(filter(lambda x: x.status, self.populations))

		insect_position_alive = [x.pos for x in self.populations]
		if self.populations:
			self.global_best = copy.deepcopy(sorted(self.populations, key=lambda x: x.fitness, reverse=True)[0])

		return insect_position_alive,insect_position_involved,insect_position_traps


class screen(object):
	def __init__(self, length, width, step):
		"""
		200*200见方的一个仓库，左上角有一些空地空地的大小为6*12
		* * * * * * * * * * * * 3 3 3 3 3 3 3 3
		* * * * * * * * * * * * 3 + + + + + + 3
		* * * * * * * * * * * * 3 + 4 4 4 + + 3
		* * * * * * * * * * * * 3 + 4 4 4 + + 3
		* * * * * * * * * * * * 3 + + + + + + 3
		* * * * * * * * * * * * 3 2 2 2 2 2 2 3
		3 3 3 3 3 3 3 3 3 3 3 3 3 + + + + + + 3
		3 + + + + + + + + + + + + + + + + + + 3
		3 + 2 2 2 2 + 4 4 + + 4 + 4 4 + 4 + + 3
		3 + 2 + 4 + + 4 + 4 + 4 + 4 4 + 4 + + 3
		3 + 2 4 4 4 + 4 + 4 4 4 4 4 4 + + + + 3
		3 + 2 + 4 + + + + 4 + 4 + 4 4 + 4 + + 3
		3 + 2 + + + + + + 4 + + + 4 4 + 4 + + 3
		3 + 2 + + + + + + 4 + + + 4 4 + + + + 3
		3 + 2 + 4 4 + 2 + 4 + 4 + + + + + + + 3
		3 + 2 4 4 4 + 2 + 4 4 4 4 2 2 + 4 + 4 3
		3 + 2 + 4 4 + 2 + 4 + 4 + + + + 4 + 4 3
		3 + + + + + + 2 + + + + 4 4 4 + 2 2 2 3
		3 + + 2 2 2 + + + + + + 4 4 4 4 + + + 3
		3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
		:param length:
		:param width:
		:param step:
		:param machine_num:
		"""
		self.x = length
		self.y = width
		self.step = step

		self.x_num = length // step
		self.y_num = width // step

		self.in_machine_num = 0
		self.in_trap_num = 0

		self.initial()

	# an interesting function...
	# self.plot()

	def eva(self, pos, eat_num):
		# when the coordinate is 200,200//10=20, which is out of the index(19)
		x_coor = min(19, int(pos[0] // self.step))
		y_coor = min(19, int(pos[1] // self.step))

		if self.food[x_coor, y_coor] >= eat_num:
			self.food[x_coor, y_coor] = self.food[x_coor, y_coor] - eat_num

		else:
			self.food[x_coor, y_coor] = 0
		return self.food[x_coor, y_coor]

	def update(self):
		"""
		the food is increasing with a fixed speed
		:return:
		"""
		for i in range(self.x_num):
			for j in range(self.y_num):
				self.food[i, j] += 0.5

		self.in_machine_num = 0

	def initial(self):
		map_1 = ['* * * * * * * * * * * * + + + + + + + +',
				 '* * * * * * * * * * * * + + q + + + + +',
				 '* * * * * * * * * * * * + + + + + + + +',
				 '* * * * * * * * * * * * + + + + + + + +',
				 '* * * * * * * * * * * * + + + + + + + +',
				 '* * * * * * * * * * * * + + + + + + + +',
				 '+ + + + + + + + + + + + + q q + + + + +',
				 '+ + + + + + + + + + + + + + + + + + + +',
				 '+ + + + + + + f f + + f + q q + + + + +',
				 '+ + + + + + + f + + + + + + + + + + + +',
				 '+ + + + m + + f + + + m + q q + + + + +',
				 '+ + + + + + + + + + + + + + + + + + + +',
				 '+ + + + + + + + + f + + + q q + + + + +',
				 '+ + + + + + + + + + + + + + + + + + + +',
				 '+ + + + + f + + + + + + + q q + + + + q',
				 '+ + + + m f + + + + + m + + + + + + f +',
				 '+ + + + + + + + + + + + + + + + + + + q',
				 '+ + + + + + + + + + + + f + + + + + + +',
				 '+ + + + + + + + + + + + + + + + + + + +',
				 '+ + + + + + + + + + + + q q q + + + + +']
		map_2 = ['* * * * * * * * * * * * 3 3 3 3 3 3 3 3',
				 '* * * * * * * * * * * * 3 + + + + + + 3',
				 '* * * * * * * * * * * * 3 + 4 4 4 + + 3',
				 '* * * * * * * * * * * * 3 + 4 4 4 + + 3',
				 '* * * * * * * * * * * * 3 + + + + + + 3',
				 '* * * * * * * * * * * * 3 2 2 2 2 2 2 3',
				 '3 3 3 3 3 3 3 3 3 3 3 3 3 + + + + + + 3',
				 '3 + + + + + + + + + + + + + + + + + + 3',
				 '3 + 2 2 2 2 + 4 4 + + 4 + 4 4 + 4 + + 3',
				 '3 + 2 + 4 + + 4 + 4 + 4 + 4 4 + 4 + + 3',
				 '3 + 2 4 4 4 + 4 + 4 4 4 4 4 4 + + + + 3',
				 '3 + 2 + 4 + + + + 4 + 4 + 4 4 + 4 + + 3',
				 '3 + 2 + + + + + + 4 + + + 4 4 + 4 + + 3',
				 '3 + 2 + + + + + + 4 + + + 4 4 + + + + 3',
				 '3 + 2 + 4 4 + 2 + 4 + 4 + + + + + + + 3',
				 '3 + 2 4 4 4 + 2 + 4 4 4 4 2 2 + 4 + 4 3',
				 '3 + 2 + 4 4 + 2 + 4 + 4 + + + + 4 + 4 3',
				 '3 + + + + + + 2 + + + + 4 4 4 + 2 2 2 3',
				 '3 + + 2 2 2 + + + + + + 4 4 4 4 + + + 3',
				 '3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3']

		self.map_dict = {}
		self.food = np.zeros((self.x_num, self.y_num))

		for i in range(self.x_num):
			temp = map_2[i].split(' ')
			for j in range(self.y_num):
				if temp[j] == '*':
					continue
				elif temp[j] == '+':
					self.food[i, j] = np.exp(max(0, np.random.normal(0, 1)))
				elif temp[j] == '2':
					self.food[i, j] = np.exp(max(0, np.random.normal(1, 1)))
				elif temp[j] == '3':
					self.food[i, j] = np.exp(max(0, np.random.normal(2, 1)))
				elif temp[j] == '4':
					self.food[i, j] = np.exp(max(0, np.random.normal(3, 1)))

		#不做广度搜索的版本（没有边际递减的）
		self.capture_prob = np.zeros((self.x_num, self.y_num))
		for i in range(self.x_num):
			temp = map_1[i].split(' ')
			for j in range(self.y_num):
				if temp[j] == 'm':
					self.capture_prob[i,j] = Parameters.prob_m
				elif temp[j] == 'f':
					self.capture_prob[i,j] = Parameters.prob_f
				elif temp[j] == 'q':
					self.capture_prob[i,j] = Parameters.prob_q

		#有广度搜索的（边际递减）
		# temp = [[0 for _ in range(self.y_num)] for _ in range(self.y_num)]
		# indexes = []
		# for i in range(self.x_num):
		# 	t = map_1[i].split(' ')
		# 	for j in range(self.y_num):
		# 		if t[j] == 'm':
		# 			temp[i][j] = 0.2
		# 			indexes.append((i, j))
		# 		elif t[j] == 'f':
		# 			temp[i][j] = 0.05
		# 			indexes.append((i, j))
		# 		elif t[j] == 'q':
		# 			temp[i][j] = 0.15
		# 			indexes.append((i, j))
		#
		# def bfs(index):
		#
		# 	queue = [index]
		# 	dic = [(0, 1), (0, -1), (1, 0), (-1, 0)]
		# 	while queue:
		# 		ii, jj = queue.pop(0)
		# 		for d in dic:
		# 			if 0 <= ii + d[0] <= 19 and 0 <= jj + d[1] <= 19 and temp[ii][jj] >= 0.02 and temp[ii][jj] - 0.02 > \
		# 					temp[ii + d[0]][jj + d[1]] and not (ii + d[0] <= 5 and jj + d[1] <= 11):
		# 				temp[ii + d[0]][jj + d[1]] = temp[ii][jj] - 0.02
		# 				queue.append((ii + d[0], jj + d[1]))
		#
		# for index in indexes:
		# 	bfs(index)
		#
		# tt = []
		# for ttt in temp:
		# 	tt.append(list(map(lambda x: round(x, 2), ttt)))
		# self.capture_prob = np.array(tt)

	def plot(self):
		vlines = np.linspace(0, 2, self.x_num + 1)
		hlines = np.linspace(0, 2, self.y_num + 1)

		plt.hlines(hlines, min(hlines), max(hlines))
		plt.vlines(vlines, min(vlines), max(vlines))

		xs, ys = np.meshgrid(vlines[1:], hlines[:-1])

		for i, (x, y) in enumerate(zip(xs.flatten(), ys.flatten()[::-1])):
			t_x = i // self.x_num
			t_y = i % self.x_num
			plt.text(x, y, self.map_dict[t_x, t_y], horizontalalignment='right', verticalalignment='bottom')
		plt.axis('off')

		plt.show()


class Individual(object):

	def __init__(self, x=None):
		self.rank = None
		self.crowding_distance = None
		self.domination_count = 0 # 这个解被支配的次数
		self.dominated_solutions = []  # 被这个解支配的解
		self.x = x
		self.objectives = None
		self.traps = []
		self.asf = 0

	def __eq__(self, other):
		if isinstance(self, other.__class__):
			return (self.x == other.x).all()
		return False

	def traps_generate(self, x_num, y_num, step):
		for i in range(x_num):
			for j in range(y_num):
				if self.x[i, j] == 1:
					self.traps.append(trap(np.array([i * step, j * step])))

	def dominates(self, other_individual):
		if self.__eq__(other_individual):
			return False

		and_condition = True
		or_condition = False
		for first, second in zip(self.objectives, other_individual.objectives):
			and_condition = and_condition and first <= second
			or_condition = or_condition or first < second
		return (and_condition and or_condition)


class populations(object):
	def __init__(self, pop_num, x_num, y_num):
		self.pop_num = pop_num
		self.pops = []
		self.fronts = []
		self.x_num = x_num
		self.y_num = y_num

		self.crossover_num = pop_num//2
		self.mutation_num = pop_num

		self.insect_population = None

		# Todo apple the surrogate model
		# self.surrogate_model = surrogate_net()
		# self.surrogate_model.load_state_dict(torch.load('surrogate_model/surrogate_model_parameters.pkl'))

	def initial(self):
		for _ in range(self.pop_num):
			temp_x = np.random.rand(self.x_num, self.y_num)
			temp_y = np.random.rand(self.x_num, self.y_num)
			temp = np.random.rand()
			# if temp<0.1:
			# 	temp_y[:].fill(0.9)
			# if 0.1<temp<0.3:
			# 	temp_y[:].fill(0.5)
			index = temp_x > temp_y
			temp_x[:].fill(0)
			temp_x[index] = 1
			temp_x[:6,:12] = 0

			self.pops.append(Individual(temp_x))
	def eva(self):
		for pop in self.pops:
			self.evaluate(pop)

	def eva_multiprocessing(self,start=None,end=None):
		if not start and not end:
			start = 0
			end =  len(self.pops)
		pool = Pool(10)


		result = []

		for i in range(start,end):
			result.append(pool.apply_async(self.evaluate_modified,args=(self.pops[i].x,i,copy.deepcopy(self.insect_population))))
		pool.close()
		pool.join()

		final_result = []
		for r in result:
			final_result.append(r.get())

		for obj1,obj2,ii in final_result:
			self.pops[ii].objectives = [obj1,obj2]





	def evaluate_modified(self,*args):
		x,i,insect_population = args
		# print('the {0}th pop is under evaluating'.format(i))

		in_machine_nums, _, _ = simulator.simulate(x, Parameters.insect_iteration,insect_population)
		probaility = [0 if not x else 1 / (2 * (1 + np.exp(-x))) for x in in_machine_nums]
		cost = [1 + Parameters.discount_q if x > Parameters.threshold else Parameters.discount_p for x in probaility]

		for index, pro in enumerate(probaility):
			if not pro:
				cost[index] = 0

		final_loss = sum(cost)
		# norm
		final_loss /= (1 + Parameters.discount_q) * Parameters.insect_iteration
		num = x.sum()/((Parameters.x/Parameters.step+1)*(Parameters.y/Parameters.step+1))

		return num,final_loss, i



	def evaluate(self, pop):
		"""
		:param pop: entity.Individual
		:return:
		"""
		in_machine_nums, _,_ = simulator.simulate(pop.x,Parameters.insect_iteration,copy.deepcopy(self.insect_population))
		probaility = [0 if not x else 1/(2*(1+np.exp(-x))) for x in in_machine_nums]

		cost = [1+Parameters.discount_q if x > Parameters.threshold else Parameters.discount_p for x in probaility]

		for index, pro in enumerate(probaility):
			if not pro:
				cost[index] = 0

		final_loss = sum(cost)
		#norm
		final_loss /= (1+Parameters.discount_q)*Parameters.insect_iteration

		pop.objectives = pop.x.sum()/(self.x_num * self.y_num), final_loss

	def fast_dominated_sort(self):
		self.fronts = [[]]
		for individual in self.pops:
			individual.domination_count = 0
			individual.dominated_solutions = []
			individual.rank = 0
			for other_individual in self.pops:
				if individual.dominates(other_individual):
					individual.dominated_solutions.append(other_individual)
				elif other_individual.dominates(individual):
					individual.domination_count += 1
			if individual.domination_count == 0:
				individual.rank = 0
				self.fronts[0].append(individual)
		# in next part the number of domination_count is changed to 0.
		i = 0
		while len(self.fronts[i]) > 0:
			temp = []
			for individual in self.fronts[i]:
				for other_individual in individual.dominated_solutions:
					other_individual.domination_count -= 1
					if other_individual.domination_count == 0:
						other_individual.rank = i + 1
						temp.append(other_individual)
			i = i + 1
			self.fronts.append(temp)

	def crowding_distance(self):
		for front in self.fronts:
			self.calculate_crowding_distance(front)

	def calculate_crowding_distance(self, front):
		if len(front) > 0:
			solutions_num = len(front)
			for individual in front:
				individual.crowding_distance = 0

			for m in range(len(front[0].objectives)):
				front.sort(key=lambda individual: individual.objectives[m])
				front[0].crowding_distance = 10 ** 9
				front[solutions_num - 1].crowding_distance = 10 ** 9
				m_values = [individual.objectives[m] for individual in front]
				scale = max(m_values) - min(m_values)
				if scale == 0: scale = 1
				for i in range(1, solutions_num - 1):
					front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[
						m]) / scale

	def pop_sort(self):
		self.pops = sorted(self.pops, key=lambda x: (x.rank, x.crowding_distance))

	def offspring_generate_modified(self):
		for _ in range(self.crossover_num):
			index = np.random.rand(self.x_num, self.y_num) < np.random.rand(self.x_num, self.y_num)
			parents1, parents2 = random.choices(self.pops, k=2)
			temp1 = parents1.x.copy()
			temp2 = parents2.x.copy()
			temp1[index] = parents2.x[index]
			temp1[:6,:12] = 0
			temp2[:6,:12] = 0
			temp2[index] = parents1.x[index]
			self.pops.append(Individual(temp1))
			self.pops.append(Individual(temp2))

		count = 0
		for pop in self.pops:
			index = np.random.rand(self.x_num, self.y_num) < np.full((self.x_num, self.y_num), 0.1)
			temp = pop.x.copy()
			temp[index] = 1 - temp[index]
			temp[:6,:12] = 0
			self.pops.append(Individual(temp))
			count += 1
			if count  == self.mutation_num:
				break
		self.eva_multiprocessing(self.pop_num,len(self.pops))

	def offspring_generate(self):
		"""
		crossover and mutation for binary variable
		:return:
		"""
		for _ in range(self.crossover_num):
			index = np.random.rand(self.x_num, self.y_num) < np.random.rand(self.x_num, self.y_num)
			parents1, parents2 = random.choices(self.pops, k=2)
			temp1 = parents1.x.copy()
			temp2 = parents2.x.copy()
			temp1[index] = parents2.x[index]
			temp2[index] = parents1.x[index]
			temp1[:6, :12] = 0
			temp2[:6, :12] = 0

			self.pops.append(Individual(temp1))
			self.evaluate(self.pops[-1])
			self.pops.append(Individual(temp2))
			self.evaluate(self.pops[-1])

		count = 0
		for pop in self.pops:
			index = np.random.rand(self.x_num, self.y_num) < np.full((self.x_num, self.y_num), 0.1)
			temp = pop.x.copy()
			temp[index] = 1 - temp[index]
			temp[:6, :12] = 0
			self.pops.append(Individual(temp))
			self.evaluate(self.pops[-1])
			count += 1
			if count  == self.mutation_num:
				break

	def update(self):
		self.pops = self.pops[:self.pop_num]

	def SOI_identify(self,ideal):
		eliminated = [False for _ in range(len(self.pops))]

		axy = self.axy_calcul(ideal)
		eliminated_num = 0
		while eliminated_num<Parameters.eliminated_number:
			ii,jj = np.unravel_index(np.argmax(axy),axy.shape)
			if not eliminated[ii] and not eliminated[jj]:
				if self.pops[ii].asf<self.pops[jj].asf:
					eliminated[jj] = True
				else:
					eliminated[ii] = True
				eliminated_num += 1

			axy[ii,jj] = 0

		SOI = []
		for index, flag in enumerate(eliminated):
			if not flag:
				SOI.append(copy.deepcopy(self.pops[index]))

		return SOI



	def axy_calcul(self,ideal):
		n = len(self.pops)

		axy = np.zeros((n,n))

		for i in range(n):
			for j in range(i+1,n):
				x_1 = self.pops[i].objectives[0]-ideal[0]
				x_2 = self.pops[i].objectives[1]-ideal[1]
				y_1 = self.pops[j].objectives[0]-ideal[0]
				y_2 = self.pops[j].objectives[1]-ideal[1]

				axy[i,j] = np.arccos((x_1*y_1)+(x_2*y_2))/min(1e-6,(np.sqrt(x_1**2+x_2**2)*np.sqrt(y_1**2+y_2**2)))
				axy[j,i] = axy[i,j]
		return axy

	def asf_calcul(self,ideal):
		for pop in self.pops:
			f_sum = sum(pop.objectives)
			w_1 = min(pop.objectives[0]/f_sum,1e-6)
			w_2 = min(pop.objectives[1]/f_sum,1e-6)
			f_1 = pop.objectives[0]-ideal[0]
			f_2 = pop.objectives[1]-ideal[1]

			pop.asf = max(f_1/w_1,f_2/w_2)





class Archive(object):
	def __init__(self,maximum):
		self.insect_population = None
		self.pops = []
		self.maximum = maximum

	def eva_modified(self):
		if not self.pops:
			return
		pool = Pool(10)

		result = []
		for i in range(len(self.pops)):
			result.append(pool.apply_async(self.evaluate_modified,args=(self.pops[i].x,i,copy.deepcopy(self.insect_population))))
		pool.close()
		pool.join()

		final_result = []
		for r in result:
			final_result.append(r.get())

		for obj1, obj2, ii,_ in final_result:
			self.pops[ii].objectives = [obj1, obj2]

	def evaluate_modified(self, *args):

		x, i, insect_population = args

		# the ith pops the jth insects
		# print('the {0}th pop in archive is under evaluating'.format(i))

		in_machine_nums, _, insects_num = simulator.simulate(x, Parameters.insect_iteration, insect_population)
		probaility = [0 if not x else 1 / (2 * (1 + np.exp(-x))) for x in in_machine_nums]
		cost = [1 + Parameters.discount_q if x > Parameters.threshold else Parameters.discount_p for x in probaility]

		for index, pro in enumerate(probaility):
			if not pro:
				cost[index] = 0

		final_loss = sum(cost)
		# norm
		final_loss /= (1 + Parameters.discount_q) * Parameters.insect_iteration
		num = x.sum() / ((Parameters.x / Parameters.step + 1) * (Parameters.y / Parameters.step + 1))

		return num, final_loss, i,insects_num

	def evaluate(self, pop):
		"""
		:param pop: entity.Individual
		:return:
		"""
		in_machine_nums, _,_ = simulator.simulate(pop.x,Parameters.insect_iteration,copy.deepcopy(self.insect_population))
		probaility = [0 if not x else 1/(2*(1+np.exp(-x))) for x in in_machine_nums]

		cost = [1+Parameters.discount_q if x > Parameters.threshold else Parameters.discount_p for x in probaility]

		for index, pro in enumerate(probaility):
			if not pro:
				cost[index] = 0

		final_loss = sum(cost)
		#norm
		final_loss /= (1+Parameters.discount_q)*Parameters.insect_iteration

		pop.objectives = pop.x.sum()/((Parameters.x//Parameters.step+1)*(Parameters.y//Parameters.step+1)), final_loss

	def update(self,SOIs,ideal):
		self.eva_modified()
		if self.pops:
			ideal[0]  = min(min([x.objectives[0] for x in self.pops]),ideal[0])
			ideal[1] = min(min([x.objectives[0] for x in self.pops]), ideal[1])

		for pop in SOIs:
			pop.m_distance = abs(pop.objectives[0]-ideal[0])+abs(pop.objectives[1]-ideal[1])


		archive_threshold = max([x.m_distance for x in SOIs])

		new_pops = []

		for pop in self.pops:
			pop.m_distance = abs(pop.objectives[0]-ideal[0])+abs(pop.objectives[1]-ideal[1])
			if pop.m_distance<archive_threshold*Parameters.alpha:
				new_pops.append(pop)

		if len(new_pops)+len(SOIs)>self.maximum:
			new_pops.sort(key = lambda x:x.m_distance)
			new_pops = new_pops[:self.maximum-len(SOIs)]

		new_pops = new_pops + SOIs

		self.pops = new_pops

		return ideal

	def fast_dominated_sort(self):
		self.fronts = [[]]
		for individual in self.pops:
			individual.domination_count = 0
			individual.dominated_solutions = []
			individual.rank = 0
			for other_individual in self.pops:
				if individual.dominates(other_individual):
					individual.dominated_solutions.append(other_individual)
				elif other_individual.dominates(individual):
					individual.domination_count += 1
			if individual.domination_count == 0:
				individual.rank = 0
				self.fronts[0].append(individual)
		# in next part the number of domination_count is changed to 0.
		i = 0
		while len(self.fronts[i]) > 0:
			temp = []
			for individual in self.fronts[i]:
				for other_individual in individual.dominated_solutions:
					other_individual.domination_count -= 1
					if other_individual.domination_count == 0:
						other_individual.rank = i + 1
						temp.append(other_individual)
			i = i + 1
			self.fronts.append(temp)

	def final_process(self):

		insect_pops = []
		for i in range(Parameters.min_insect_num,Parameters.max_insect_num+1):
			insect_pops.append(insect_population(i,screen(Parameters.x,Parameters.y,Parameters.step)))


		pool = Pool(12)
		result = []
		for j in range(len(insect_pops)):
			for i in range(len(self.pops)):
				result.append(pool.apply_async(self.evaluate_modified,args=(self.pops[i].x,i,copy.deepcopy(insect_pops[j]))))
		pool.close()
		pool.join()

		finalresult = []
		for r in result:
			finalresult.append(r.get())

		objectives = [[0,0] for _ in range(len(self.pops))]
		insect_num = [[] for _ in range(len(self.pops))]

		for f in finalresult:
			objectives[f[2]][0] += f[0]
			objectives[f[2]][1] += f[1]
			insect_num[f[2]].append(f[3])
		scenario_num = Parameters.max_insect_num-Parameters.min_insect_num+1
		for i in range(len(self.pops)):
			self.pops[i].objectives[0] = objectives[i][0]/scenario_num
			self.pops[i].objectives[1] = objectives[i][1] / scenario_num
			self.pops[i].insect_num = insect_num[i]

		self.fast_dominated_sort()

		self.fronts[0].sort(key = lambda x:x.objectives[0])
		count = 0
		for pop in self.fronts[0]:
			for temp in pop.insect_num:
				plt.plot(range(len(temp)),temp)
			plt.savefig('png/'+str(count)+'.png')
			plt.show()
			count += 1












if __name__ == '__main__':
	pass
