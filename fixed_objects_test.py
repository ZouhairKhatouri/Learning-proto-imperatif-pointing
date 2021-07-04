### Imports:

import numpy as np
from math import pi, atan
import matplotlib.pyplot as plt
from classes.arm import Arm
from classes.cargiver import CareGiver
from random import sample
import pickle as pkl
import json

### Configuration :

with open("env_config.json", "rb") as file:
    env_config = json.load(file)

M = env_config["M"]
N_itt = env_config["N_itt"]
N_exp = env_config["N_exp"]
N_pitt = env_config["N_pitt"]
batch_size = env_config["batch_size"]

nu = env_config["nu"]

N_objs = env_config['N_objs']

N_anodes = env_config["N_anodes"]
N_snodes = env_config["N_snodes"]

dcr = env_config["dcr"]
des = env_config["des"]
dx = env_config["dx"]
D = env_config["D"]

### Experiments :

Results = {'Dirac':np.zeros((N_exp, N_itt - M)),\
            'GaussianS':np.zeros((N_exp, N_itt - M)),\
            'GaussianWS':np.zeros((N_exp, N_itt - M))}

for expr in range(N_exp):

    objs = sample([i-20 for i in range(10)], N_objs)
    theta_d = [atan(obj/D) + pi/2 for obj in objs]
    env_config['theta_d'] = theta_d

    # Dirac kernel :

    print("Experiment - %s - %i"%("Dirac kernel", expr))

    env_config['nu'] = 0
    cg = CareGiver(env_config)
    arm = Arm(env_config)

    slide = []

    for itt in range(N_itt):

        cg.T = 3*pi/N_anodes

        arm.decided_wanted_obj()
        action = arm.take_next_action()
        R, _ = cg.give_reward(action, arm.wanted_obj)
        arm.update_inner_state(action, R, kernel=None)

        if len(slide) == M:
            Results['Dirac'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

    # Gaussian kernel with scaffolding :

    print("Experiment - %s - %i"%("Gaussian kernel with scaffolding", expr))

    env_config['nu'] = nu
    cg = CareGiver(env_config)
    arm = Arm(env_config)

    slide = []

    for itt in range(N_itt):

        arm.decided_wanted_obj()
        action = arm.take_next_action()
        R, _ = cg.give_reward(action, arm.wanted_obj)
        arm.update_inner_state(action, R, kernel='gaussian')

        if len(slide) == M:
            Results['GaussianS'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

    # Gaussian kernel without scaffolding :

    print("Experiment - %s - %i"%("Gaussian kernel without scaffolding", expr))

    env_config['nu'] = 0
    cg = CareGiver(env_config)
    arm = Arm(env_config)

    slide = []

    for itt in range(N_itt):

        cg.T = 3*pi/N_anodes

        arm.decided_wanted_obj()
        action = arm.take_next_action()
        R, _ = cg.give_reward(action, arm.wanted_obj)
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