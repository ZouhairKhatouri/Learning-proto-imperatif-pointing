### Imports:

import numpy as np
from math import pi, atan
import matplotlib.pyplot as plt
from classes.eyeAndArm import EyeAndArm
from classes.cargiver import CareGiver
from random import sample
import pickle as pkl
import torch
import torch.nn as nn
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

mse = nn.MSELoss()

Results = {'GaussianS':np.zeros((N_exp, N_itt - M)),\
            'GaussianWS':np.zeros((N_exp, N_itt - M))}

for expr in range(N_exp):

    # With scaffolding :

    objs = sample(list(np.linspace(-D/dcr, D/dcr, N_snodes)), N_objs)

    print("Experiment - Moving object - %s - %i"%("With scaffolding", expr))

    env_config["nu"] = nu
    eaa = EyeAndArm(env_config)

    for pitt in range(N_pitt):

        a = np.zeros((N_anodes, N_anodes))
        a[tuple(np.array(sample(range(N_anodes), 2)))] = 1
        a = torch.from_numpy(a).float()
        try:
            dr = eaa.get_hand_on_retina(a)
        except Exception:
            continue
        a_hat = eaa.mc(a)
        dr_hat = eaa.mc.encode(a)

        loss = 0.5*mse(a.view(-1, N_anodes, N_anodes), a_hat) + 0.5*mse(dr.view(1, -1), dr_hat)
        loss.backward()
        eaa.optimizer.step()
        if pitt % batch_size == 0:
            eaa.optimizer.zero_grad()
            if pitt == 0:
                loss_0 = loss.item()
                print('Pointing learning loss {}/{}: {}'.format(pitt, N_pitt, loss.item()))
            else:
                print('Pointing learning loss {}/{}: {}%'.format(pitt, N_pitt, 100*(loss.item() - loss_0)/loss_0))

    eaa.update_retina_to_action_mapper()
        
    slide = []

    for itt in range(N_itt):

        theta_d = [atan((obj - des)/D) + pi/2 for obj in objs]
        env_config["theta_d"] = theta_d
        cg = CareGiver(env_config)
        abstract_retina = np.array([[obj, objs[obj]] for obj in range(N_objs)])

        eaa.decided_wanted_obj()
        action = eaa.take_next_action()
        R = cg.give_reward(action, eaa.wanted_obj)
        eaa.update_inner_state(action, R, abstract_retina)

        if len(slide) == M:
            Results['GaussianS'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

        objs[0] += dx

'''    # Without scaffolding :

    objs = sample(list(np.linspace(-D/dcr, D/dcr, N_snodes)), N_objs)

    print("Experiment - Moving object - %s - %i"%("Without scaffolding", expr))

    env_config["nu"] = 0
    eaa = EyeAndArm(env_config)

    for pitt in range(N_pitt):

        a = np.zeros((N_anodes, N_anodes))
        a[tuple(np.array(sample(range(N_anodes), 2)))] = 1
        a = torch.from_numpy(a).float()
        try:
            dr = eaa.get_hand_on_retina(a)
        except Exception:
            continue
        a_hat = eaa.mc(a)
        dr_hat = eaa.mc.encode(a)

        loss = 0.5*mse(a.view(-1, N_anodes, N_anodes), a_hat) + 0.5*mse(dr.view(1, -1), dr_hat)
        loss.backward()
        eaa.optimizer.step()
        if pitt % batch_size == 0:
            eaa.optimizer.zero_grad()
            if pitt == 0:
                loss_0 = loss.item()
                print('Pointing learning loss {}/{}: {}'.format(pitt, N_pitt, loss.item()))
            else:
                print('Pointing learning loss {}/{}: {}%'.format(pitt, N_pitt, 100*(loss.item() - loss_0)/loss_0))

    eaa.update_retina_to_action_mapper()

    slide = []

    for itt in range(N_itt):

        cg.T = 3*pi/N_anodes # No scaffolding

        theta_d = [atan(obj/D) + pi/2 for obj in objs]
        env_config["theta_d"] = theta_d
        cg = CareGiver(env_config)
        abstract_retina = np.array([[obj, objs[obj]] for obj in range(N_objs)])

        eaa.decided_wanted_obj()
        action = eaa.take_next_action()
        R = cg.give_reward(action, eaa.wanted_obj)
        eaa.update_inner_state(action, R, abstract_retina)

        if len(slide) == M:
            Results['GaussianWS'][expr, itt - M] = np.mean(slide)
            slide.pop(0)
        slide.append(R)

        objs[0] += dx'''

plt.axhline(y = 1, color = 'green', linestyle = '-')
#plt.plot(range(N_itt - M), Results['GaussianWS'].mean(axis = 0), color = 'red', label='Gaussian kernel, without scaffolding')
#plt.fill_between(range(N_itt - M), Results['GaussianWS'].mean(axis = 0)-Results['GaussianWS'].std(axis = 0), Results['GaussianWS'].mean(axis = 0)+Results['GaussianWS'].std(axis = 0),color='gray', alpha=0.2)
plt.plot(range(N_itt - M), Results['GaussianS'].mean(axis = 0), color = 'blue', label='Gaussian kernel, with scaffolding')
plt.fill_between(range(N_itt - M), Results['GaussianS'].mean(axis = 0)-Results['GaussianS'].std(axis = 0), Results['GaussianS'].mean(axis = 0)+Results['GaussianS'].std(axis = 0),color='gray', alpha=0.2)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Mean reward over %i last epochs"%(M))
plt.show()

with open('dumped/moving_object_jacobien_transfer_model_2_N_obj_1_N_exp_10.obj', 'wb') as file: 
    pkl.dump(Results, file)