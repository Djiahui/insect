import numpy as np
import entity
import matplotlib.pyplot as plt
import copy



def draw(pops,traps,i):
    poses = list(map(lambda x:x.pos,pops.populations))
    poses = np.vstack(poses)

    trap_pos = list(map(lambda x:x.pos,traps))
    trap_pos = np.vstack(trap_pos)
    plt.scatter(poses[:,0],poses[:,1])
    plt.scatter(trap_pos[:,0],trap_pos[:,1],s=5000,c='r',alpha=0.5)
    plt.title(str(i))
    plt.xticks(np.arange(0,100,10))
    plt.yticks(np.arange(0,100,10))
    plt.grid()
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

    for i in range(iteration):
        pops.update(traps)
        print(pops.global_best.fitness)
        draw(pops,traps,i)

    return sum(pops.env.food)


def matrix_generate(x, y, step):
    x_num = x//step +1
    y_num = y//step +1

    matrix = [[0]*y_num for _ in range(x_num)]

    for i in range(x_num):
        for j in range(y_num):
            r1 = np.random.rand()
            r2 = np.random.rand()
            if r2<r1:
                matrix[i][j] = 1
    return matrix




def sample_generate(sample_num,env,pops,insect_iteration):
    data = {}

    for _ in range(sample_num):
        matrix = matrix_generate(env.x,env.y,env.step)
        new_pops = copy.deepcopy(pops)
        food_rest = simulate(matrix,insect_iteration,new_pops)
        food_loss = (sum(pops.env.food)-food_rest)/sum(pops.env.food)

        n = len(data)
        data[n] = {}
        data[n]['sample'] = matrix
        data[n]['label'] = food_loss

    return data








if __name__ == '__main__':
    pass


    pop_num = 100
    trap_num = 5

    iteration = 20
    x,y = 100,100
    step = 10

    env = entity.screen(x,y,step)
    pops = entity.insect_population(pop_num, env)


    # simulate(pop_num,matrix,iteration)