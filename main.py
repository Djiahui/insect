import entity
import numpy as np
import nds
import matplotlib.pyplot as plt
import simulator


def main(x,y,step,insect_num,sample_num,insect_iteration):
	env = entity.screen(x, y, step)
	pops = entity.insect_population(insect_num, env)
	simulator.sample_generate(sample_num,env,pops,insect_iteration)
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
	x = 100
	y = 100
	step = 10
	insect_num = 1000
	sample_num = 1000
	insect_iteration = 50
	main(x,y,step,insect_num,sample_num,insect_iteration)
