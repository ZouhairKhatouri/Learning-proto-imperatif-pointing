### Imports:

import numpy as np
from math import pi, atan
import matplotlib.pyplot as plt
from classes.arm import Arm
from classes.cargiver import CareGiver
from random import sample
import pickle as pkl

### Configuration :

RANDOM_CHOICE = False
M = 50 # Smoothing parameter
N_nodes = 100
N_exp = 100
epsilon = 0.0
h = 2
D = 10
N_objs = 1
threshold = 0.7

env_conf = {}
env_conf['RANDOM_CHOICE'] = RANDOM_CHOICE
env_conf['N_nodes'] = N_nodes
env_conf['epsilon'] = epsilon
env_conf['h'] = h
env_conf['N_objs'] = N_objs

### Experiments :

Results = np.zeros((N_exp, 101))

for nu in np.linspace(0, 1, 101):

    print("Experiments with nu={}".format(nu))

    env_conf['nu'] = nu

    for expr in range(N_exp):

        objs = sample([i-20 for i in range(10)], N_objs)
        theta_d = [atan(obj/D) + pi/2 for obj in objs]
        env_conf['theta_d'] = theta_d

        cg = CareGiver(env_conf)
        arm = Arm(env_conf)

        slide = []
        itt = 1

        while True:

            arm.decided_wanted_obj()
            action = arm.take_next_action()
            R = cg.give_reward(action, arm.wanted_obj)
            arm.update_inner_state(action, R, kernel='gaussian')

            if len(slide) == M:
                if np.mean(slide) > threshold:
                    Results[expr, int(nu*100)] = itt
                    break
                slide.pop(0)
            slide.append(R)

            itt += 1

plt.plot(np.linspace(0, 1, 101), Results.mean(axis = 0), color = 'red')
plt.fill_between(np.linspace(0, 1, 101), Results.mean(axis = 0)-Results.std(axis = 0), Results.mean(axis = 0)+Results.std(axis = 0),color='gray', alpha=0.2)
plt.xlabel("nu")
plt.ylabel("Mean number of epochs to get at least {} mean reward".format(threshold))
plt.show()

with open('dumped/schaffolding_param_0_7_11.obj', 'wb') as file: 
    pkl.dump(Results, file)