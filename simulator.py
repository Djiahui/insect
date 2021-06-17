import numpy as np
import entity
import matplotlib.pyplot as plt



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






def simulate(pop_num,trap_num,iteration):
    env = entity.screen(100, 100)
    pops = entity.insect_population(pop_num,env)
    traps = []
    for i in range(trap_num):
        traps.append(entity.trap(env))

    pops.generate(traps)

    for i in range(iteration):
        pops.update(traps)
        print(pops.global_best.fitness)
        draw(pops,traps,i)








if __name__ == '__main__':


    pop_num = 100
    trap_num = 5

    iteration = 20

    simulate(pop_num,trap_num,iteration)