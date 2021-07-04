### Imports:

import numpy as np
from math import pi, atan
import matplotlib.pyplot as plt
from classes.retinaAndArm import RetinaAndArm
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

Results = {'Mode 0':np.zeros((N_exp, N_itt - M)),\
            'Mode 1':np.zeros((N_exp, N_itt - M)),\
            'd 0':np.zeros((N_exp, N_itt - M)),\
            'd 1':np.zeros((N_exp, N_itt - M))}

for expr in range(N_exp):

    # Mode 0 :

    env_config["SCAFFOLDING_MODE"] = 0

    objs = sample(list(np.linspace(-D/dcr, D/dcr, N_snodes)), N_objs)

    print("Experiment - %s - %i"%("Mode 0", expr))

    eaa = RetinaAndArm(env_config)

    '''running_loss = 0.0

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

        eaa.optimizer.zero_grad()
        loss = 0.5*torch.norm(a.view(-1, N_anodes, N_anodes) - a_hat) + 0.5*torch.norm(dr.view(1, -1) - dr_hat)
        loss.backward()
        eaa.optimizer.step()

        running_loss += loss.item()
        if pitt % batch_size == 0:
            if pitt == 0:
                loss_0 = running_loss
                print('Pointing learning loss {}/{}: {}'.format(pitt, N_pitt, loss_0))
            else:
                print('Pointing learning loss {}/{}: {}%'.format(pitt, N_pitt, 100*(running_loss/batch_size - loss_0)/loss_0))
                running_loss = 0.0

    eaa.update_retina_to_action_mapper()'''
        
    slide_R = []
    slide_d = []

    for itt in range(N_itt):

        theta_d = [atan((obj - des)/D) + pi/2 for obj in objs]
        env_config["theta_d"] = theta_d
        cg = CareGiver(env_config)
        abstract_retina = np.array([[obj, objs[obj]] for obj in range(N_objs)])

        eaa.decided_wanted_obj()
        action = eaa.take_next_action()
        R, d = cg.give_reward(action, eaa.wanted_obj)
        eaa.update_inner_state(action, R, abstract_retina)

        if len(slide_R) == M:
            Results['Mode 0'][expr, itt - M] = np.mean(slide_R)
            slide_R.pop(0)
        slide_R.append(R)

        if len(slide_d) == M:
            Results['d 0'][expr, itt - M] = np.mean(slide_d)
            slide_d.pop(0)
        slide_d.append(d)

    # Mode 1 :

    env_config["SCAFFOLDING_MODE"] = 1

    objs = sample(list(np.linspace(-D/dcr, D/dcr, N_snodes)), N_objs)

    print("Experiment - %s - %i"%("Mode 1", expr))

    eaa = RetinaAndArm(env_config)

    '''running_loss = 0.0

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

        eaa.optimizer.zero_grad()
        loss = 0.5*torch.norm(a.view(-1, N_anodes, N_anodes) - a_hat) + 0.5*torch.norm(dr.view(1, -1) - dr_hat)
        loss.backward()
        eaa.optimizer.step()

        running_loss += loss.item()
        if pitt % batch_size == 0:
            if pitt == 0:
                loss_0 = running_loss
                print('Pointing learning loss {}/{}: {}'.format(pitt, N_pitt, loss_0))
            else:
                print('Pointing learning loss {}/{}: {}%'.format(pitt, N_pitt, 100*(running_loss/batch_size - loss_0)/loss_0))
                running_loss = 0.0

    eaa.update_retina_to_action_mapper()'''
        
    slide_R = []
    slide_d = []

    for itt in range(N_itt):

        theta_d = [atan((obj - des)/D) + pi/2 for obj in objs]
        env_config["theta_d"] = theta_d
        cg = CareGiver(env_config)
        abstract_retina = np.array([[obj, objs[obj]] for obj in range(N_objs)])

        eaa.decided_wanted_obj()
        action = eaa.take_next_action()
        R, d = cg.give_reward(action, eaa.wanted_obj)
        eaa.update_inner_state(action, R, abstract_retina)

        if len(slide_R) == M:
            Results['Mode 1'][expr, itt - M] = np.mean(slide_R)
            slide_R.pop(0)
        slide_R.append(R)

        if len(slide_d) == M:
            Results['d 1'][expr, itt - M] = np.mean(slide_d)
            slide_d.pop(0)
        slide_d.append(d)

plt.axhline(y = 1, color = 'green', linestyle = '-')
plt.plot(range(N_itt - M), Results['Mode 0'].mean(axis = 0), color = 'red', label='Mode 0')
#plt.fill_between(range(N_itt - M), Results['Mode 0'].mean(axis = 0)-Results['Mode 0'].std(axis = 0), Results['Mode 0'].mean(axis = 0)+Results['Mode 0'].std(axis = 0),color='gray', alpha=0.2)
plt.plot(range(N_itt - M), Results['Mode 1'].mean(axis = 0), color = 'blue', label='Mode 1')
#plt.fill_between(range(N_itt - M), Results['Mode 1'].mean(axis = 0)-Results['Mode 1'].std(axis = 0), Results['Mode 1'].mean(axis = 0)+Results['Mode 1'].std(axis = 0),color='gray', alpha=0.2)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Mean reward over %i last epochs"%(M))
plt.show()

plt.clf()

plt.plot(range(N_itt - M), Results['d 0'].mean(axis = 0), color = 'red', label='Mode 0')
#plt.fill_between(range(N_itt - M), Results['Mode 0'].mean(axis = 0)-Results['Mode 0'].std(axis = 0), Results['Mode 0'].mean(axis = 0)+Results['Mode 0'].std(axis = 0),color='gray', alpha=0.2)
plt.plot(range(N_itt - M), Results['d 1'].mean(axis = 0), color = 'blue', label='Mode 1')
#plt.fill_between(range(N_itt - M), Results['Mode 1'].mean(axis = 0)-Results['Mode 1'].std(axis = 0), Results['Mode 1'].mean(axis = 0)+Results['Mode 1'].std(axis = 0),color='gray', alpha=0.2)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Mean quadratique error over %i last epochs"%(M))
plt.show()

with open('dumped/scaffolding_comparison_N_obj_3_N_exp_10.obj', 'wb') as file: 
    pkl.dump(Results, file)