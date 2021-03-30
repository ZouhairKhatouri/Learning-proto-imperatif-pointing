### Imports:

import numpy as np
from math import pi, atan
import matplotlib.pyplot as plt
from arm import Arm
from cargiver import CareGiver
from random import sample
import pickle as pkl

### Configuration :

RANDOM_CHOICE = False
M = 50 # Smoothing parameter
N_itt = 4000 + M
N_nodes = 100
N_exp = 100
epsilon = 0.0
h = 2
nu = 0.3
D = 10
N_objs = 5

env_conf = {}
env_conf['RANDOM_CHOICE'] = RANDOM_CHOICE
env_conf['N_nodes'] = N_nodes
env_conf['epsilon'] = epsilon
env_conf['h'] = h
env_conf['N_objs'] = N_objs

### Experiments :

Results = {'Dirac':np.zeros((N_exp, N_itt - M)),\
            'GaussianS':np.zeros((N_exp, N_itt - M)),\
            'GaussianWS':np.zeros((N_exp, N_itt - M))}

for expr in range(N_exp):

    objs = sample([i-20 for i in range(10)], N_objs)
    theta_d = [atan(obj/D) + pi/2 for obj in objs]
    env_conf['theta_d'] = theta_d

    # Dirac kernel :

    print("Experiment - %s - %i"%("Dirac kernel", expr))

    env_conf['nu'] = 0
    cg = CareGiver(env_conf)
    arm = Arm(env_conf)

    slide = []

    for itt in range(N_itt):

        cg.T = 3*pi/N_nodes

        arm.decided_wanted_obj()
        action = arm.take_next_action()
        R = cg.give_reward(action, arm.wanted_obj)
        arm.update_inner_state(action, R, kernel=None)

        if len(slide) == M:
            Results['Dirac'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

    # Gaussian kernel with scaffolding :

    print("Experiment - %s - %i"%("Gaussian kernel with scaffolding", expr))

    env_conf['nu'] = nu
    cg = CareGiver(env_conf)
    arm = Arm(env_conf)

    slide = []

    for itt in range(N_itt):

        arm.decided_wanted_obj()
        action = arm.take_next_action()
        R = cg.give_reward(action, arm.wanted_obj)
        arm.update_inner_state(action, R, kernel='gaussian')

        if len(slide) == M:
            Results['GaussianS'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

    # Gaussian kernel without scaffolding :

    print("Experiment - %s - %i"%("Gaussian kernel without scaffolding", expr))

    env_conf['nu'] = 0
    cg = CareGiver(env_conf)
    arm = Arm(env_conf)

    slide = []

    for itt in range(N_itt):

        cg.T = 3*pi/N_nodes

        arm.decided_wanted_obj()
        action = arm.take_next_action()
        R = cg.give_reward(action, arm.wanted_obj)
        arm.update_inner_state(action, R, kernel='gaussian')

        if len(slide) == M:
            Results['GaussianWS'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

plt.axhline(y = 1, color = 'green', linestyle = '-')
plt.plot(range(N_itt - M), Results['Dirac'].mean(axis = 0), color = 'black', label='Dirac kernel')
#plt.fill_between(range(N_itt - M), Results['Dirac'].mean(axis = 0)-Results['Dirac'].std(axis = 0), Results['Dirac'].mean(axis = 0)+Results['Dirac'].std(axis = 0),color='gray', alpha=0.2)
plt.plot(range(N_itt - M), Results['GaussianWS'].mean(axis = 0), color = 'red', label='Gaussian kernel, without scaffolding')
#plt.fill_between(range(N_itt - M), Results['GaussianWS'].mean(axis = 0)-Results['GaussianWS'].std(axis = 0), Results['GaussianWS'].mean(axis = 0)+Results['GaussianWS'].std(axis = 0),color='gray', alpha=0.2)
plt.plot(range(N_itt - M), Results['GaussianS'].mean(axis = 0), color = 'blue', label='Gaussian kernel, with scaffolding')
#plt.fill_between(range(N_itt - M), Results['GaussianS'].mean(axis = 0)-Results['GaussianS'].std(axis = 0), Results['GaussianS'].mean(axis = 0)+Results['GaussianS'].std(axis = 0),color='gray', alpha=0.2)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Mean reward over %i last epochs"%(M))
plt.show()

with open('dumped/main_4000.obj', 'wb') as file: 
    pkl.dump(Results, file)