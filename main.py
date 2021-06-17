import entity
import numpy as np
import nds
import matplotlib.pyplot as plt


def main():
	insect_num = 10
	pop_num = 10
	env = entity.screen(100, 100,10)


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
	main()
