import copy
import pickle

import entity
import numpy as np
import nds
import matplotlib.pyplot as plt
import simulator
from parameters import Parameters


def main(x, y, step, insect_num, sample_num, insect_iteration, pop_num, train=True):
	if train:
		simulator.sample_generate(x, y, step, insect_num, sample_num, insect_iteration)

	exit()
	optimize(pop_num)


def optimize(pop_num):

	population = entity.populations(pop_num,21,21)




	insect_pops = entity.insect_population(Parameters.get_random_insect_number(),entity.screen(Parameters.x,Parameters.y,Parameters.step))
	population.insect_population = insect_pops
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
	# main(Parameters.x, Parameters.y, Parameters.step, Parameters.insect_num, Parameters.sample_num, Parameters.insect_iteration, Parameters.pop_num)
	optimize(Parameters.pop_num)