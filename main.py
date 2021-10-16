import copy
import pickle

import entity
import numpy as np
import nds
import matplotlib.pyplot as plt
import simulator


def main(x,y,step,insect_num,sample_num,insect_iteration):
	env = entity.screen(x, y, step,4)
	pops = entity.insect_population(insect_num, copy.deepcopy(env))
	simulator.sample_generate(sample_num,env,copy.deepcopy(pops),insect_iteration)
	exit()


	population = entity.populations(pop_num,env,entity.insect_population(insect_num,env))
	population.initial()
	for _ in range(10):
		population.offspring_generate()
		population.fast_dominated_sort()
		population.crowding_distance()
		population.pop_sort()
		population.update()
		draw(population)

	exit()




def draw(population):
	temp = list(map(lambda x:x.objectives,population.pops))
	temp = np.array(temp)

	plt.scatter(temp[:,0],temp[:,1])
	plt.show()








if __name__ == '__main__':
	x = 200
	y = 200
	step = 10
	insect_num = 1000
	sample_num = 1000
	insect_iteration = 50
	main(x,y,step,insect_num,sample_num,insect_iteration)
