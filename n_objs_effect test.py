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
N_exp = 10
epsilon = 0.0
h = 2
nu = 0.3
D = 10
N_objs_max = 20
threshold = 0.1

env_conf = {}
env_conf['RANDOM_CHOICE'] = RANDOM_CHOICE
env_conf['N_nodes'] = N_nodes
env_conf['epsilon'] = epsilon
env_conf['h'] = h

### Experiments :

Results = {'Dirac':np.zeros((N_exp, N_objs_max)),\
            'GaussianS':np.zeros((N_exp, N_objs_max)),\
            'GaussianWS':np.zeros((N_exp, N_objs_max))}

for N_objs in range(1, N_objs_max+1):

    print("Experiments with {} objects".format(N_objs))

    env_conf['N_objs'] = N_objs

    for expr in range(N_exp):

        # Dirac kernel :

        objs = sample([i-2*N_objs_max for i in range(N_objs_max)], N_objs)
        theta_d = [atan(obj/D) + pi/2 for obj in objs]
        env_conf['theta_d'] = theta_d

        env_conf['nu'] = 0
        cg = CareGiver(env_conf)
        arm = Arm(env_conf)

        slide = []
        itt = 1

        while True:

            cg.T = 3*pi/N_nodes

            arm.decided_wanted_obj()
            action = arm.take_next_action()
            R = cg.give_reward(action, arm.wanted_obj)
            arm.update_inner_state(action, R, kernel=None)

            if len(slide) == M:
                if np.mean(slide) > threshold:
                    Results['Dirac'][expr, N_objs-1] = itt
                    break
                slide.pop(0)
            slide.append(R)

            itt += 1

        # Gaussian kernel with scaffolding :

        env_conf['nu'] = nu
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
                    Results['GaussianS'][expr, N_objs-1] = itt
                    break
                slide.pop(0)
            slide.append(R)

            itt += 1

        # Gaussian kernel without scaffolding :

        env_conf['nu'] = 0
        cg = CareGiver(env_conf)
        arm = Arm(env_conf)

        slide = []
        itt = 1 

        while True:

            cg.T = 3*pi/N_nodes

            arm.decided_wanted_obj()
            action = arm.take_next_action()
            R = cg.give_reward(action, arm.wanted_obj)
            arm.update_inner_state(action, R, kernel='gaussian')

            if len(slide) == M:
                if np.mean(slide) > threshold:
                    Results['GaussianWS'][expr, N_objs-1] = itt
                    break
                slide.pop(0)
            slide.append(R)

            itt += 1

plt.plot(range(1, N_objs_max+1), Results['Dirac'].mean(axis = 0), color = 'black', label='Dirac kernel')
#plt.fill_between(range(1, N_objs_max+1), Results['Dirac'].mean(axis = 0)-Results['Dirac'].std(axis = 0), Results['Dirac'].mean(axis = 0)+Results['Dirac'].std(axis = 0),color='gray', alpha=0.2)
plt.plot(range(1, N_objs_max+1), Results['GaussianWS'].mean(axis = 0), color = 'red', label='Gaussian kernel, without scaffolding')
#plt.fill_between(range(1, N_objs_max+1), Results['GaussianWS'].mean(axis = 0)-Results['GaussianWS'].std(axis = 0), Results['GaussianWS'].mean(axis = 0)+Results['GaussianWS'].std(axis = 0),color='gray', alpha=0.2)
plt.plot(range(1, N_objs_max+1), Results['GaussianS'].mean(axis = 0), color = 'blue', label='Gaussian kernel, with scaffolding')
#plt.fill_between(range(1, N_objs_max+1), Results['GaussianS'].mean(axis = 0)-Results['GaussianS'].std(axis = 0), Results['GaussianS'].mean(axis = 0)+Results['GaussianS'].std(axis = 0),color='gray', alpha=0.2)
plt.legend()
plt.xlabel("Number of objects of interest")
plt.ylabel("Mean number of epochs to get at least {} mean reward".format(threshold))
plt.show()

with open('dumped/n_objs_effect_0.1.obj', 'wb') as file: 
    pkl.dump(Results, file)