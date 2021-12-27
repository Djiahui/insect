import copy
import os
import pickle

import entity
import numpy as np
import nds
import matplotlib.pyplot as plt
import simulator
from parameters import Parameters
import time
from multiprocessing import Process, Pool

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def main(x, y, step, insect_num, sample_num, insect_iteration, pop_num, train=True):
	if train:
		simulator.sample_generate(x, y, step, insect_num, sample_num, insect_iteration)

	exit()
	optimize(pop_num)


def ideal_update(ideal,temp):
	temp_ideal = [min([x.objectives[0] for x in temp.pops]),min([x.objectives[1] for x in temp.pops])]
	return [min(ideal[0],temp_ideal[0]),min(temp_ideal[1],ideal[1])]


def optimize(pop_num):

	population = entity.populations(pop_num,Parameters.x//Parameters.step+1,Parameters.y//Parameters.step+1)
	insect_pops = entity.insect_population(Parameters.get_random_insect_number(),entity.screen(Parameters.x,Parameters.y,Parameters.step))
	population.insect_population = insect_pops
	population.initial()
	ideal = [1,1]

	archive = entity.Archive(Parameters.archive_maximum)
	archive.insect_population = insect_pops
	for iter in range(Parameters.iteration):
		print('the {} iteration'.format(iter))
		print(time.strftime("%H:%M:%S")+': evaluate with new insect poopulation')
		population.eva_multiprocessing()
		print(time.strftime("%H:%M:%S")+': generate new pops')
		population.offspring_generate_modified()
		population.fast_dominated_sort()
		population.crowding_distance()
		population.pop_sort()
		ideal = ideal_update(ideal, population)
		SOI = population.SOI_identify(ideal)
		print(time.strftime("%H:%M:%S")+': archive update')
		ideal = archive.update(SOI,ideal)

		population.update()

		insect_pops = entity.insect_population(Parameters.get_random_insect_number(),entity.screen(Parameters.x,Parameters.y,Parameters.step))
		population.insect_population = insect_pops
		archive.insect_population = insect_pops

		draw_traps(archive,iter)

		# if not iter%10:
		# 	draw(archive.pops)
	print(time.strftime("%H:%M:%S")+': final process')
	archive.final_process()
	draw_pareto_front(archive.fronts[0])

	print('save the result')
	final_objectives = [x.objectives for x in archive.fronts[0]]
	final_decision = [x.x for x in archive.fronts[0]]
	if os.path.exists('final_objectives'):
		os.remove('final_objectives')
	with open('final_objectives', 'wb') as pkl:
		pickle.dump(final_objectives, pkl)
	with open('final_decsion','wb') as pkl2:
		pickle.dump(final_decision,pkl2)




def draw_pareto_front(pops,i=-1):
	temp = [x.objectives for x in pops]
	temp = np.array(temp)

	plt.scatter(temp[:, 0], temp[:, 1])
	if i==-1:
		plt.title('Pareto Optimal Solutions')
		plt.savefig('png/Pareto Optimal Solutions.png')
	else:
		plt.title('{}th iteration archive'.format(i))
		plt.savefig('png/{}th iteration archive.png'.format(i))
	plt.show()

def draw_traps(archive,iteration):

	archive.fast_dominated_sort()
	draw_pareto_front(archive.fronts[0],iteration)
	n = len(archive.fronts[0])
	for i in range(n):
		archive.fronts[0][i].traps_generate(Parameters.x//Parameters.step+1,Parameters.y//Parameters.step+1,Parameters.step)

		trap_pos = list(map(lambda x: [x.pos[1], 200 - x.pos[0]], archive.fronts[0][i].traps))
		trap_pos = np.vstack(trap_pos)
		plt.scatter(trap_pos[:, 0], trap_pos[:, 1], c='r', alpha=0.5)
	# plt.scatter(trap_pos[:,0],trap_pos[:,1],s=2000,c='r',alpha=0.5)
		plt.grid()
		plt.xticks(np.arange(0, 210, 10))
		plt.yticks(np.arange(0, 210, 10))
		plt.title('{}th iteration'.format(iteration))
		plt.savefig('png/{}th iteration {}th figure.png'.format(iteration,i))
		plt.show()




def draw_2():
	with open('final_objectives','rb') as pkl:
		temp = pickle.load(pkl)

	temp.sort()
	temp = np.array(temp[:-1])
	plt.scatter(temp[:,0], temp[:,1])
	plt.show()

if __name__ == '__main__':
	# main(Parameters.x, Parameters.y, Parameters.step, Parameters.insect_num, Parameters.sample_num, Parameters.insect_iteration, Parameters.pop_num)
	optimize(Parameters.pop_num)