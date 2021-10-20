import random

import numpy as np
import entity
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

import pickle
import os

def draw(pops,traps):
    """
    # the Coordinate transformation is applied such as [0,10]--[10,200]
    :param pops:
    :param traps:
    :return:
    """
    poses = list(map(lambda x:[x.pos[1],200-x.pos[0]],pops.populations))
    poses = np.vstack(poses)

    trap_pos = list(map(lambda x:[x.pos[1],200-x.pos[0]],traps))
    trap_pos = np.vstack(trap_pos)
    plt.scatter(poses[:,0],poses[:,1])
    plt.scatter(trap_pos[:, 0], trap_pos[:, 1], c='r', alpha=0.5)
    # plt.scatter(trap_pos[:,0],trap_pos[:,1],s=2000,c='r',alpha=0.5)
    # plt.title(str(i))
    plt.grid()
    plt.xticks(np.arange(0,200,10))
    plt.yticks(np.arange(0,200,10))
    plt.show()


def trap_generate(matrix,x,y,step):
    x_num = x//step + 1
    y_num = y//step + 1
    traps = []
    for i in range(x_num):
        for j in range(y_num):
            if matrix[i][j]:
                traps.append(entity.trap([i*step,j*step]))

    return traps



def simulate(matrix,iteration,pops):
    traps = trap_generate(matrix,pops.env.x,pops.env.y,pops.env.step)
    pops.generate(traps)
    draw(pops,traps)

    with open('data_from_sim/temperature_data','rb') as pkl:
        temp_data = pickle.load(pkl)

    day_count = 0
    for _,temp in temp_data.items():
        if day_count>iteration:
            break
        pops.update(traps,temp)
        draw(pops, traps)

        day_count += 1

    exit()


    return pops.env.food.sum()


def matrix_generate(x, y, step,test=True):
    x_num = x//step +1
    y_num = y//step +1

    matrix = [[0]*y_num for _ in range(x_num)]

    if test:
        for i in range(x_num):
            # left to right
            for j in range(y_num):
                #from bottom to top
                # only for current screen
                if i<6 and j<12:
                    continue
                if random.random()<0.05:
                    matrix[i][j] = 1
        return matrix
    temp = np.random.rand()
    if temp<0.25:
        for i in range(x_num):
            for j in range(y_num):
                # only for current screen
                if i>15 and j<12:
                    continue
                r1 = np.random.rand()
                if r1<0.1:
                    matrix[i][j] = 1
    elif 0.25<temp<0.5:
        for i in range(x_num):
            for j in range(y_num):
                # only for current screen
                if i>15 and j<12:
                    continue
                r1 = np.random.rand()
                if r1<0.2:
                    matrix[i][j] = 1
    elif 0.5<temp<0.75:
        for i in range(x_num):
            for j in range(y_num):
                # only for current screen
                if i>15 and j<12:
                    continue
                r1 = np.random.rand()
                if r1<0.25:
                    matrix[i][j] = 1
    else:
        for i in range(x_num):
            for j in range(y_num):
                # only for current screen
                if i>15 and j<12:
                    continue
                r1 = np.random.rand()
                r2 = np.random.rand()
                if r2 < r1:
                    matrix[i][j] = 1
    return matrix




def sample_generate(sample_num,env,pops,insect_iteration):


    # for _ in range(sample_num):
    for _ in tqdm(range(sample_num)):
        if not os.path.exists('surrogate model\data2.pkl'):
            data = {}
            with open('surrogate model\data2.pkl','wb') as pkl:
                pickle.dump(data,pkl)
        with open('surrogate model\data2.pkl','rb') as pkl1:
            data = pickle.load(pkl1)

        matrix = matrix_generate(env.x,env.y,env.step)
        new_pops = copy.deepcopy(pops)
        food_rest = simulate(matrix,insect_iteration,new_pops)
        food_loss = (pops.env.food.sum()-food_rest)/pops.env.food.sum()

        n = len(data)
        data[n] = {}
        data[n]['sample'] = matrix
        data[n]['label'] = food_loss

        with open('data2.pkl', 'wb') as pkl:
            pickle.dump(data, pkl)

    # Todo attention !
    # exit()




    return data










if __name__ == '__main__':

    pop_num = 100
    trap_num = 5

    iteration = 20
    x,y = 100,100
    step = 10

    env = entity.screen(x,y,step)
    pops = entity.insect_population(pop_num, env)


    # simulate(pop_num,matrix,iteration)