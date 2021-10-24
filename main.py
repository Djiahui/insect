import copy
import pickle

import entity
import numpy as np
import nds
import matplotlib.pyplot as plt
import simulator


def main(x, y, step, insect_num, sample_num, insect_iteration, pop_num, train=True):
	if train:
		simulator.sample_generate(x, y, step, insect_num, sample_num, insect_iteration)

	exit()
	optimize(pop_num)


def optimize(pop_num):
	population = entity.populations(pop_num,21,21)
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
	temp = list(map(lambda x: x.objectives, population.pops))
	temp = np.array(temp)

	plt.scatter(temp[:, 0], temp[:, 1])
	plt.show()


if __name__ == '__main__':
	x = 200
	y = 200
	step = 10
	insect_num = 10
	sample_num = 1000
	insect_iteration = 10
	pop_num = 10
	main(x, y, step, insect_num, sample_num, insect_iteration, pop_num)
