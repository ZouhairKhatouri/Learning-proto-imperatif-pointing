### Imports:

import numpy as np
from math import pi, atan
import matplotlib.pyplot as plt
from eyeAndArm import EyeAndArm
from cargiver import CareGiver
from random import sample
import pickle as pkl

### Configuration :

RANDOM_CHOICE = False
M = 50 # Smoothing parameter
N_itt = 4000 + M
N_nodes = 100
N_exp = 10
epsilon = 0.0
h = 2
nu = 0.3
D = 10
N_objs = 1
dx = 0.1 # Change in the position of the moving object

env_conf = {}
env_conf['RANDOM_CHOICE'] = RANDOM_CHOICE
env_conf['N_nodes'] = N_nodes
env_conf['epsilon'] = epsilon
env_conf['h'] = h
env_conf['N_objs'] = N_objs
env_conf['D'] = D

### Experiments :

Results = {'GaussianS':np.zeros((N_exp, N_itt - M)),\
            'GaussianWS':np.zeros((N_exp, N_itt - M))}

for expr in range(N_exp):

    # With scaffolding :

    objs = sample([i-20 for i in range(10)], N_objs)

    print("Experiment - Moving object - %s - %i"%("With scaffolding", expr))

    env_conf['nu'] = nu
    eaa = EyeAndArm(env_conf)

    slide = []

    for itt in range(N_itt):

        theta_d = [atan(obj/D) + pi/2 for obj in objs]
        env_conf['theta_d'] = theta_d
        cg = CareGiver(env_conf)
        Retina = np.array([[obj, objs[obj]] for obj in range(N_objs)])

        eaa.decided_wanted_obj()
        action = eaa.take_next_action()
        R = cg.give_reward(action, eaa.wanted_obj)
        eaa.update_inner_state(action, R, Retina)

        if len(slide) == M:
            Results['GaussianS'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

        objs[0] += dx

    # Without scaffolding :

    objs = sample([i-20 for i in range(10)], N_objs)

    print("Experiment - Moving object - %s - %i"%("Without scaffolding", expr))

    env_conf['nu'] = 0
    eaa = EyeAndArm(env_conf)

    slide = []

    for itt in range(N_itt):

        cg.T = 3*pi/N_nodes

        theta_d = [atan(obj/D) + pi/2 for obj in objs]
        env_conf['theta_d'] = theta_d
        cg = CareGiver(env_conf)
        Retina = np.array([[obj, objs[obj]] for obj in range(N_objs)])

        eaa.decided_wanted_obj()
        action = eaa.take_next_action()
        R = cg.give_reward(action, eaa.wanted_obj)
        eaa.update_inner_state(action, R, Retina)

        if len(slide) == M:
            Results['GaussianWS'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

        objs[0] += dx

plt.axhline(y = 1, color = 'green', linestyle = '-')
plt.plot(range(N_itt - M), Results['GaussianWS'].mean(axis = 0), color = 'red', label='Gaussian kernel, without scaffolding')
#plt.fill_between(range(N_itt - M), Results['GaussianWS'].mean(axis = 0)-Results['GaussianWS'].std(axis = 0), Results['GaussianWS'].mean(axis = 0)+Results['GaussianWS'].std(axis = 0),color='gray', alpha=0.2)
plt.plot(range(N_itt - M), Results['GaussianS'].mean(axis = 0), color = 'blue', label='Gaussian kernel, with scaffolding')
#plt.fill_between(range(N_itt - M), Results['GaussianS'].mean(axis = 0)-Results['GaussianS'].std(axis = 0), Results['GaussianS'].mean(axis = 0)+Results['GaussianS'].std(axis = 0),color='gray', alpha=0.2)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Mean reward over %i last epochs"%(M))
plt.show()

with open('dumped/moving_object_4000.obj', 'wb') as file: 
    pkl.dump(Results, file)