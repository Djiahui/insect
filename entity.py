import numpy as np
import copy

class insect(object):
	def __init__(self, pos=None, direction=None, rate=None):
		if pos:
			self.pos = pos
		else:
			self.pos = np.random.rand(2)*100

		if direction:
			self.direction = direction
		else:
			self.direction = np.random.rand(2)
			self.direction = self.direction / np.linalg.norm(self.direction, ord=2)

		if rate:
			self.rate = rate
		else:
			self.rate = 1

		self.best_pos = np.zeros(2)
		self.best_fit = None
		self.fitness = None
		self.status = True

		self.eat_num = 0.5

	def update(self, global_best, env):

		temp1 = self.best_pos - self.pos
		temp2 = global_best.pos - self.pos

		self.direction = temp1 + temp2
		if self.direction.sum():
			self.direction = self.direction / np.linalg.norm(self.direction, ord=2)

		self.pos = self.pos + self.direction * self.rate

		if self.pos[0] < 0:
			self.pos[0] = -self.pos[0]
		if self.pos[0] > env.x:
			self.pos[0] = 2 * env.x - self.pos[0]

		if self.pos[1] < 0:
			self.pos[1] = -self.pos[1]
		if self.pos[1] > env.y:
			self.pos[1] = 2 * env.y - self.pos[1]

		# Todo 应该先判断死活在看吃粮与否

		self.fitness = env.eva(self.pos,self.eat_num)
		if self.fitness > self.best_fit:
			self.best_fit = self.fitness
			self.best_pos = self.pos

	def living_test(self,traps):
		for trap in traps:
			dis = np.linalg.norm(self.pos-trap.pos,ord=2)

			if dis<trap.radius:
				self.status = False
				break






class trap(object):
	def __init__(self, pos):
		self.pos = pos
		self.radius = 3


class insect_population(object):
	def __init__(self, insect_num, env):
		self.insect_num = insect_num
		self.populations = []

		self.global_best = None
		self.env = env
		self.dead_num = 0


	def generate(self,traps,num=None):
		if num ==None:
			num = self.insect_num

		temp_num = 0
		while temp_num<num:
			temp = insect()
			temp.fitness = self.env.eva(temp.pos,temp.eat_num)
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
					self.global_best = temp

	def update(self,traps):
		for temp in self.populations:
			temp.update(self.global_best, self.env)
			temp.living_test(traps)
			# if not temp.status:
			# 	print('one')
			# 	continue
		self.global_best = sorted(self.populations,key=lambda x:x.fitness,reverse=True)[0]

		# add a function to generate new population


		self.populations = list(filter(lambda x:x.status,self.populations))
		short_num = self.insect_num-len(self.populations)
		self.dead_num += short_num
		# print(len(self.populations))
		self.generate(traps,short_num)
		# print(len(self.populations))


class screen(object):
	def __init__(self, length, width,step):
		self.x = length
		self.y = width
		self.step = step

		self.x_num = length//step
		self.y_num = width//step

		self.food_init()

	def food_init(self):
		self.food = 50+np.random.rand(self.x_num,self.y_num)*50

	def eva(self, pos,eat_num):
		x_coor = int(pos[0]//self.step)
		y_coor = int(pos[1]//self.step)

		if self.food[x_coor,y_coor]>=eat_num:
			self.food[x_coor,y_coor] = self.food[x_coor,y_coor]-eat_num

		else:
			self.food[x_coor,y_coor] = 0
		return self.food[x_coor,y_coor]


class Individual(object):

	def __init__(self,x=None):
		self.rank = None
		self.crowding_distance = None
		self.domination_count = None   #这个解被支配的次数
		self.dominated_solutions = None   #被这个解支配的解
		self.x = x
		self.objectives = None
		self.traps = []

	def __eq__(self, other):
		if isinstance(self, other.__class__):
			return (self.x == other.x).all()
		return False

	def traps_generate(self,x_num,y_num,step):
		for i in range(x_num):
			for j in range(y_num):
				if self.x[i,j]==1:
					self.traps.append(trap(np.array([x_num*step,y_num*step])))


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
	def __init__(self,pop_num,screen,insect_pop):
		self.pop_num = pop_num
		self.pops = []
		self.fronts = []
		self.screen = screen
		self.insect_pop = insect_pop

	def initial(self):
		for _ in range(self.pop_num):
			temp_x = np.random.rand(self.screen.x_num,self.screen.y_num)
			temp_x[temp_x>0.5] = 1
			temp_x[temp_x<=0.5] = 0
			insects = copy.deepcopy(self.insect_pop)
			self.pops.append(Individual(temp_x))
			self.pops[-1].traps_generate(self.screen.x_num,self.screen.y_num,self.screen.step)
			self.pops[-1].objectives = self.evaluate(self.pops[-1].traps,insects)


	def evaluate(self,traps,insect_pop):
		for _ in range(100):
			insect_pop.update(traps)

		return 1/insect_pop.dead_num,len(traps)/(self.screen.y_num*self.screen.x_num)


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

	def calculate_crowding_distance(self,front):
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
		self.pops = sorted(self.pops,key=lambda x:(x.rank,x.crowding_distance))


	def offspring_generate(self):
		for _ in range(self.pop_num):
			temp_x = np.random.rand(self.screen.x_num, self.screen.y_num)
			temp_x[temp_x > 0.5] = 1
			temp_x[temp_x <= 0.5] = 0
			insects = copy.deepcopy(self.insect_pop)
			temp = Individual(temp_x)
			temp.traps_generate(self.screen.x_num,self.screen.y_num,self.screen.step)
			temp.objectives = self.evaluate(temp.traps,insects)
			self.pops.append(temp)

	def update(self):
		self.pops = self.pops[:self.pop_num]











if __name__ == '__main__':
	pass
