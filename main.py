import copy
import os
import pickle
import random

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

def picture():
	# 数量相同 分布不同   时间的影响  数量分布相同，决策不同  决策相同，数量分布都不同
	print('final picture')
	os.makedirs('png/temp')
	with open('new_final_decision','rb') as pkl:
		temp = pickle.load(pkl)
	population = entity.populations(len(temp), Parameters.x // Parameters.step + 1, Parameters.y // Parameters.step + 1)
	insect_pops = entity.insect_population(13, entity.screen(Parameters.x, Parameters.y, Parameters.step))
	population.insect_population = insect_pops
	for pop in temp:
		population.pops.append(entity.Individual(pop))
	population.picture(1)


	insect_pops2 = entity.insect_population(13,
										   entity.screen(Parameters.x, Parameters.y, Parameters.step))
	population.insect_population = insect_pops2
	population.picture(2)

	insect_pops3 = entity.insect_population(13, entity.screen(Parameters.x, Parameters.y, Parameters.step))
	population.insect_population = insect_pops3
	population.picture(3)
	exit(0)

def optimize(pop_num):

	population = entity.populations(pop_num,Parameters.x//Parameters.step+1,Parameters.y//Parameters.step+1)
	insect_pops = entity.insect_population(13,entity.screen(Parameters.x,Parameters.y,Parameters.step))
	population.insect_population = insect_pops
	population.initial()
	ideal = [1,1]

	archive = entity.Archive(Parameters.archive_maximum)
	archive.insect_population = insect_pops
	for iter in range(Parameters.iteration):
		print('the {} iteration'.format(iter))
		print(time.strftime("%H:%M:%S")+': evaluate with new insect poopulation')
		population.eva_multiprocessing()



		population.offspring_generate_modified()
		population.fast_dominated_sort()
		population.crowding_distance()
		population.pop_sort()
		ideal = ideal_update(ideal, population)


		print('identify')
		SOI = population.SOI_identify(ideal)
		print(time.strftime("%H:%M:%S")+': archive update')
		ideal = archive.update(SOI,ideal)

		population.update()

		insect_pops = entity.insect_population(13,entity.screen(Parameters.x,Parameters.y,Parameters.step))
		population.insect_population = insect_pops
		archive.insect_population = insect_pops

		draw_traps(population,iter)

		# if not iter%10:
		# 	draw(archive.pops)
	print(time.strftime("%H:%M:%S")+': final process')
	archive.final_process()
	draw_traps(archive)

	print('save the result')
	final_objectives = [x.objectives for x in archive.fronts[0]]
	final_decision = [x.x for x in archive.fronts[0]]
	with open('new_final_objectives', 'wb') as pkl:
		pickle.dump(final_objectives, pkl)
	with open('new_final_decision','wb') as pkl2:
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

def draw_traps(archive,iteration=-1):

	# archive.fast_dominated_sort()
	draw_pareto_front(archive.fronts[0],iteration)
	archive.fronts[0].sort(key = lambda x: x.x.sum())
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
		if iteration == -1:
			plt.savefig('png/{} final figure.png'.format(i))
		else:
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
	optimize(Parameters.pop_num)
	picture()