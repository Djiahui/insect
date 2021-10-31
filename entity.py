import matplotlib.pyplot as plt
import numpy
import numpy as np
import copy
import random

import torch
from data_from_sim.predict_net import pre_net
from surrogate_model.surrogate_model import surrogate_net
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

		self.best_pos = np.zeros(2)
		self.best_fit = None
		self.fitness = None
		self.status = True
		self.stand_in_same_place = 0

		self.eat_num = 0.1

	def update(self, global_best, env, traps):

		temp1 = self.best_pos - self.pos
		temp2 = global_best.pos - self.pos

		pre_pos = [self.pos[0] // env.step, self.pos[1] // env.step]
		self.direction = self.direction+random.random()*temp1+random.random()*temp2

		self.pos = self.pos + np.linalg.norm(self.direction, ord=2)

		# self.direction = np.linalg.norm(temp1 + temp2, ord=2)+self.direction
		# if self.direction.sum():
		# 	self.pos = self.pos + self.direction * self.rate
		# 	self.direction = self.direction / np.linalg.norm(self.direction, ord=2)

		# if random.random()<0.01:
		# 	print('direction')
		# 	print(self.direction)





		#for current screen
		if 0<self.pos[0]<60 and 0<self.pos[1]<120:
			if pre_pos[0] < 6:
				self.pos[1] = 120
			else:
				self.pos[0] = 60

		self.pos[0] = min(env.x,self.pos[0])
		self.pos[0] = max(self.pos[0],0)
		self.pos[1] = min(env.y,self.pos[1])
		self.pos[1] = max(0,self.pos[1])

		# if self.pos[0] < 0:
		# 	self.pos[0] = -self.pos[0]
		# if self.pos[0] > env.x:
		# 	self.pos[0] = 2 * env.x - self.pos[0]
		#
		# if self.pos[1] < 0:
		# 	self.pos[1] = -self.pos[1]
		# if self.pos[1] > env.y:
		# 	self.pos[1] = 2 * env.y - self.pos[1]



		cur_pos = [self.pos[0] // env.step, self.pos[1] // env.step]
		self.stand_in_same_place += (1 if pre_pos == cur_pos else 0)
		#Todo 5 days
		if self.stand_in_same_place >= 5:
			self.status = False
			env.in_machine_num += 1
			return
		#
		self.living_test(traps)
		if not self.status:
			return

		self.in_machine(env)
		if not self.status:
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

	def in_machine(self, env):
		dist = 1e6
		for machine in env.machines:
			dist = min(dist, numpy.sqrt((self.pos[0] - machine.x) ** 2 + (self.pos[1] - machine.y) ** 2))

		if dist < env.machines[0].threshold:
			self.status = False if random.random() < 0.9 else True
			env.in_machine_num += 1
		else:
			self.status = False if random.random() < 0.2 else True


class trap(object):
	def __init__(self, pos):
		self.pos = pos
		self.radius = 3


class Machine(object):
	def __init__(self, x, y,step):
		"""
		the parameter x and y are coordinate of block, thus the true coordinate of machine need to be calculated based on the step parameter.
		:param x:
		:param y:
		:param step:
		"""
		self.coordinate_x = x
		self.coordinate_y = y
		self.threshold = 5
		self.x = x*step+step//2
		self.y = y*step+step//2





class insect_population(object):
	def __init__(self, insect_num, env):
		self.insect_num = insect_num
		self.populations = []

		self.global_best = None
		self.env = env
		self.dead_num = 0

		self.regression_model = pre_net()
		self.regression_model.load_state_dict(torch.load('data_from_sim/regression_model_parameters.pkl'))

	def generate(self, traps, num=None):
		if num == None:
			num = self.insect_num

		temp_num = 0
		while temp_num < num:
			temp = insect(self.env.x, self.env.y)
			if temp.pos[0]<60 and temp.pos[1]<120:
				continue
			temp.fitness = self.env.eva(temp.pos, temp.eat_num)
			temp.best_pos = temp.pos
			temp.best_fit = temp.fitness
			temp.living_test(traps)
			if not temp.status:
				continue

			self.populations.append(temp)
			temp_num += 1

			if not self.global_best:
				self.global_best = temp
			else:
				if temp.fitness > self.global_best.fitness:
					self.global_best = copy.deepcopy(temp)

	def update(self, traps,temp):
		"""
		first generate new insects based on the current(yesterday) population
		second update the position
		third living test, machine test
		:param traps:
		:return:
		"""

		current_num = len(self.populations)
		if not current_num:
			return

		input = torch.tensor([current_num]+temp)
		predict_num = round(self.regression_model(input).item())
		to_generate_num = max(0,predict_num-current_num)
		self.generate(traps,to_generate_num)

		for temp in self.populations:
			temp.update(self.global_best, self.env, traps)

		self.populations = list(filter(lambda x: x.status, self.populations))
		if self.populations:
			self.global_best = copy.deepcopy(sorted(self.populations, key=lambda x: x.fitness, reverse=True)[0])


class screen(object):
	def __init__(self, length, width, step):
		"""
		200*200见方的一个仓库，左上角有一些空地空地的大小为6*12
		* * * * * * * * * * * * + + + + + + + +
		* * * * * * * * * * * * + + q + + + + +
		* * * * * * * * * * * * + + + + + + + +
		* * * * * * * * * * * * + + + + + + + +
		* * * * * * * * * * * * + + + + + + + +
		* * * * * * * * * * * * + + + + + + + +
		+ + + + + + + + + + + + + q q + + + + +
		+ + + + + + + + + + + + + + + + + + + +
		+ + + + + + + f f + + f + q q + + + + +
		+ + + + + + + f + + + + + + + + + + + +
		+ + + + m + + f + + + m + q q + + + + +
		+ + + + + + + + + + + + + + + + + + + +
		+ + + + + + + + + f + + + q q + + + + +
		+ + + + + + + + + + + + + + + + + + + +
		+ + + + + f + + + + + + + q q + + + + q
		+ + + + m f + + + + + m + + + + + + f +
		+ + + + + + + + + + + + + + + + + + + q
		+ + + + + + + + + + + + f + + + + + + +
		+ + + + + + + + + + + + + + + + + + + +
		+ + + + + + + + + + + + q q q + + + + +
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

		self.coordinate_deter()
		self.food_init()
		self.machine_init()

		# an interesting function...
		# self.plot()


	def food_init(self):
		self.food = np.zeros((self.x_num,self.y_num))
		for i in range(self.x_num):
			for j in range(self.y_num):
				if self.map_dict[i,j] == '*':
					#no food
					continue
				elif self.map_dict[i,j] == 'm':
					# machine
					self.food[i,j] = np.random.random()*10
				elif self.map_dict[i,j] == 'q':
					#circle
					self.food[i,j] = np.random.random()*5
				else:
					#square
					self.food[i,j] = np.random.random()*2


	def eva(self, pos, eat_num):
		# when the coordinate is 200,200//10=20, which is out of the index(19)
		x_coor = min(19, int(pos[0] // self.step))
		y_coor = min(19, int(pos[1] // self.step))

		if self.food[x_coor, y_coor] >= eat_num:
			self.food[x_coor, y_coor] = self.food[x_coor, y_coor] - eat_num

		else:
			self.food[x_coor, y_coor] = 0
		return self.food[x_coor, y_coor]

	def machine_init(self):
		"""
		randomly generate some position to set machine
		:return:
		"""
		self.machines = []
		for x,y in self.machines_coordinates:
			self.machines.append(Machine(x, y,self.step))

	def update(self):
		"""
		the food is increasing with a fixed speed
		:return:
		"""
		for i in range(self.x_num):
			for j in range(self.y_num):
				self.food[i, j] += random.random() * 2 + 1

		self.in_machine_num = 0



	def coordinate_deter(self):
		map = ['* * * * * * * * * * * * + + + + + + + +',
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

		self.map_dict = {}
		self.machines_coordinates = []
		for i in range(self.x_num):
			temp = map[i].split(' ')
			for j in range(self.y_num):
				self.map_dict[(i,j)] = temp[j]
				if self.map_dict[(i,j)] == 'm':
					self.machines_coordinates.append((i,j))

	def plot(self):
		vlines = np.linspace(0,2,self.x_num+1)
		hlines = np.linspace(0,2,self.y_num+1)

		plt.hlines(hlines,min(hlines),max(hlines))
		plt.vlines(vlines,min(vlines),max(vlines))

		xs, ys = np.meshgrid(vlines[1:],hlines[:-1])

		for i,(x,y) in enumerate(zip(xs.flatten(),ys.flatten()[::-1])):
			t_x = i//self.x_num
			t_y = i%self.x_num
			plt.text(x,y,self.map_dict[t_x,t_y],horizontalalignment='right',verticalalignment='bottom')
		plt.axis('off')

		plt.show()







class Individual(object):

	def __init__(self, x=None):
		self.rank = None
		self.crowding_distance = None
		self.domination_count = None  # 这个解被支配的次数
		self.dominated_solutions = None  # 被这个解支配的解
		self.x = x
		self.objectives = None
		self.traps = []

	def __eq__(self, other):
		if isinstance(self, other.__class__):
			return (self.x == other.x).all()
		return False

	def traps_generate(self, x_num, y_num, step):
		for i in range(x_num):
			for j in range(y_num):
				if self.x[i, j] == 1:
					self.traps.append(trap(np.array([x_num * step, y_num * step])))

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
	def __init__(self, pop_num,x_num,y_num):
		self.pop_num = pop_num
		self.pops = []
		self.fronts = []
		self.x_num = x_num
		self.y_num = y_num

		self.crossover_num = pop_num
		self.mutation_num = pop_num

		#Todo apple the surrogate model
		self.surrogate_model = surrogate_net()
		self.surrogate_model.load_state_dict(torch.load('surrogate_model/surrogate_model_parameters.pkl'))

	def initial(self):
		for _ in range(self.pop_num):
			temp_x = np.random.rand(self.x_num, self.y_num)
			temp_y = np.random.rand(self.x_num,self.y_num)
			index = temp_x>temp_y
			temp_x[:].fill(0)
			temp_x[index] = 1

			self.pops.append(Individual(temp_x))
			self.evaluate(self.pops[-1])

	def evaluate(self, pop):
		"""
		:param pop: entity.Individual
		:return:
		"""
		#Todo transoforme the np array to tensor
		input = pop.x
		pop.objective = sum(pop.x),self.surrogate_model(input)


	def fast_dominated_sort(self):
		self.fronts = [[]]
		for individual in self.pops:
			individual.domination_count = 0
			individual.dominated_solutions = []
			for other_individual in self.pops:
				if individual.dominates(other_individual):
					individual.dominated_solutions.append(other_individual)
				elif other_individual.dominates(individual):
					individual.domination_count += 1
			if individual.domination_count == 0:
				individual.rank = 0
				self.fronts[0].append(individual)
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

	def offspring_generate(self):
		"""
		crossover and mutation for binary variable
		:return:
		"""
		for _ in range(self.crossover_num):
			index = np.random.rand(self.x_num,self.y_num)<np.random.rand(self.x_num,self.y_num)
			parents1,parents2 = random.choices(self.pops, k=2)
			temp1 = parents1.x[:]
			temp2 = parents2.x[:]
			temp1[index] = parents2[index]
			temp2[index] = parents1[index]
			self.pops.append(Individual(temp1))
			self.evaluate(self.pops[-1])
			self.pops.append(Individual(temp2))
			self.evaluate(self.pops[-1])

		for pop in self.pops:
			index = np.random.rand(self.x_num,self.y_num)<np.full((self.x_num,self.y_num),0.1)
			temp = pop.x[:]
			temp[index] = 1-temp[index]
			self.pops.append(Individual(temp))
			self.evaluate(self.pops[-1])

	def update(self):
		self.pops = self.pops[:self.pop_num]


if __name__ == '__main__':
	pass
