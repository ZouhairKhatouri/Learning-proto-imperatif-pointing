### Imports:

import numpy as np
from math import pi, atan
import matplotlib.pyplot as plt
from classes.vcortexAndArm import VCortexAndArm
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

nu = env_config["nu"]

N_objs = env_config['N_objs']

N_anodes = env_config["N_anodes"]
N_snodes = env_config["N_snodes"]

dcr = env_config["dcr"]
des = env_config["des"]
dx = env_config["dx"]
D = env_config["D"]

### Experiments :

Results = {'GaussianS':np.zeros((N_exp, N_itt - M)),\
            'GaussianWS':np.zeros((N_exp, N_itt - M))}

for expr in range(N_exp):

    # With scaffolding :

    objs = sample(list(np.linspace(-D/dcr, D/dcr, N_snodes+2))[1:-1], N_objs)
    s = 1

    print("Experiment - Moving object - %s - %i"%("With scaffolding", expr))

    env_config["nu"] = nu
    vaa = VCortexAndArm(env_config)
        
    slide = []

    for itt in range(N_itt):

        theta_d = [atan((obj - des)/D) + pi/2 for obj in objs]
        env_config["theta_d"] = theta_d
        cg = CareGiver(env_config)
        abstract_retina = np.array([[obj, objs[obj]] for obj in range(N_objs)])

        vaa.decided_wanted_obj()
        action = vaa.take_next_action(abstract_retina)
        R, _ = cg.give_reward(action, vaa.wanted_obj)
        vaa.update_inner_state(action, R, abstract_retina)

        if len(slide) == M:
            Results['GaussianS'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

        if objs[0] > D/dcr - dx or objs[0] < -D/dcr + dx:
            s = -s
        objs[0] += s*dx

'''    # Without scaffolding :

    objs = sample(list(np.linspace(-D/dcr, D/dcr, N_snodes+2))[1:-1], N_objs)
    s = 1

    print("Experiment - Moving object - %s - %i"%("Without scaffolding", expr))

    env_config["nu"] = 0
    vaa = VCortexAndArm(env_config)

    slide = []

    for itt in range(N_itt):

        cg.T = 3*pi/N_anodes # No scaffolding

        theta_d = [atan(obj/D) + pi/2 for obj in objs]
        env_config["theta_d"] = theta_d
        cg = CareGiver(env_config)
        abstract_retina = np.array([[obj, objs[obj]] for obj in range(N_objs)])

        vaa.decided_wanted_obj()
        action = vaa.take_next_action(abstract_retina)
        R, _ = cg.give_reward(action, vaa.wanted_obj)
        vaa.update_inner_state(action, R, abstract_retina)

        if len(slide) == M:
            Results['GaussianWS'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

        if objs[0] > D/dcr - dx or objs[0] < -D/dcr + dx:
            s = -s
        objs[0] += s*dx'''

plt.axhline(y = 1, color = 'green', linestyle = '-')
'''plt.plot(range(N_itt - M), Results['GaussianWS'].mean(axis = 0), color = 'red', label='Gaussian kernel, without scaffolding')
plt.fill_between(range(N_itt - M), Results['GaussianWS'].mean(axis = 0)-Results['GaussianWS'].std(axis = 0), Results['GaussianWS'].mean(axis = 0)+Results['GaussianWS'].std(axis = 0),color='gray', alpha=0.2)'''
plt.plot(range(N_itt - M), Results['GaussianS'].mean(axis = 0), color = 'blue', label='Gaussian kernel, with scaffolding')
plt.fill_between(range(N_itt - M), Results['GaussianS'].mean(axis = 0)-Results['GaussianS'].std(axis = 0), Results['GaussianS'].mean(axis = 0)+Results['GaussianS'].std(axis = 0),color='gray', alpha=0.2)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Mean reward over %i last epochs"%(M))
plt.show()

with open('dumped/moving_object_model_3_N_obj_1_N_exp_10.obj', 'wb') as file: 
    pkl.dump(Results, file)