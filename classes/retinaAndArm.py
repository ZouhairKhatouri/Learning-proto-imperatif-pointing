import numpy as np
from math import sqrt, pi, exp, sin, cos
from copy import deepcopy
import torch
from torch import nn

class MotorCortex(nn.Module):

    def __init__(self, N_anodes, N_rnodes):
        super(MotorCortex, self).__init__()
        self.N_anodes = N_anodes
        self.N_rnodes = N_rnodes
        self.fc_1 = nn.Linear(N_anodes**2, N_rnodes, bias=False)
        self.fc_2 = nn.Linear(N_rnodes, N_anodes**2, bias=False)

    def encode(self, a):
        return self.fc_1(a.view(-1, self.N_anodes**2))

    def decode(self, dr):
        return self.fc_2(dr).view(-1, self.N_anodes, self.N_anodes)

    def forward(self, a):
        return self.decode(self.encode(a))

class RetinaAndArm:

    previous_retina = None

    def __init__(self, env_config):

        self.RANDOM_CHOICE = env_config['RANDOM_CHOICE']

        self.epsilon = env_config['epsilon']
        self.h = env_config['h']

        self.N_objs = env_config['N_objs']

        self.N_anodes = env_config['N_anodes']
        self.N_rnodes = env_config['N_rnodes']

        self.dcr = env_config['dcr']
        self.des = env_config['des']
        self.D = env_config['D']

        # Initialisation nulle, choix au hasard
        if self.RANDOM_CHOICE:
            self.internal_map = np.zeros((self.N_objs, self.N_anodes, self.N_anodes)) 

        # Initialisation au hasard, choix déterministe
        if not self.RANDOM_CHOICE:
            self.internal_map = np.random.randn(self.N_objs, self.N_anodes, self.N_anodes)

        self.wanted_obj = None
        self.std_e = 7*pi/self.N_anodes 
        self.std_i = 4*pi/self.N_anodes 

        self.hand_on_retina_filter = np.array([[\
                self.dcr*((self.des-cos(((1-2*self.epsilon)*pi/self.N_anodes)*theta_1+self.epsilon*pi)\
                    -cos(((1-2*self.epsilon)*pi/self.N_anodes)*theta_2+self.epsilon*pi))\
                        /(sin(((1-2*self.epsilon)*pi/self.N_anodes)*theta_1+self.epsilon*pi)\
                            +sin(((1-2*self.epsilon)*pi/self.N_anodes)*theta_2+self.epsilon*pi)))\
                for theta_2 in range(self.N_anodes)]\
                for theta_1 in range(self.N_anodes)])
        self.mc = MotorCortex(self.N_anodes, self.N_rnodes)
        self.optimizer = torch.optim.Adam(self.mc.parameters(), lr=0.01)
        self.f = (lambda dr: torch.zeros(1, self.N_anodes, self.N_anodes))

    def decided_wanted_obj(self):
        self.wanted_obj = np.random.choice(range(self.N_objs))

    def take_next_action(self):
        return self.infer_action(self.internal_map[self.wanted_obj,:,:])

    def infer_action(self, some_map):
        # Initialisation nulle, choix au hasard
        if self.RANDOM_CHOICE:
            return tuple(((1-2*self.epsilon)*pi/self.N_anodes)*t+self.epsilon*pi for t in np.unravel_index(np.random.choice(np.flatnonzero(some_map == some_map.max())), (self.N_anodes, self.N_anodes)))
        # Initialisation au hasard, choix déterministe
        if not self.RANDOM_CHOICE:
            return tuple(((1-2*self.epsilon)*pi/self.N_anodes)*t+self.epsilon*pi for t in np.unravel_index(some_map.argmax(), (self.N_anodes, self.N_anodes)))

    def update_inner_state(self, action, R, abstract_retina):

        if self.previous_retina is None:
            self.previous_retina =  self.abstract_to_concrete_retina(abstract_retina)

        dr = self.abstract_to_concrete_retina(abstract_retina) - self.previous_retina
        da = self.infer_action(self.f(dr))
        #da = [0, 0]

        # Building the kernel :
        N = -(((np.linspace(self.epsilon*pi,(1-self.epsilon)*pi,self.N_anodes) - (action[0] + da[0]))**2).reshape((self.N_anodes, 1))\
            + (np.linspace(self.epsilon*pi,(1-self.epsilon)*pi,self.N_anodes) - (action[1] + da[1]))**2)/2
        if R > 0:
            N /= self.std_e**2
            N = np.exp(N)/(self.std_i*sqrt(2*pi))
        else:
            N /= self.std_i**2
            N = np.exp(N)/(self.std_i*sqrt(2*pi))

        # Applying the kernel :
        self.internal_map[self.wanted_obj, :, :] = ((self.h - 1)/self.h)*self.internal_map[self.wanted_obj, :, :] + R*N

    def point_on_retina(self, x, y):
        r = np.zeros((self.N_rnodes,))
        p = int(self.N_rnodes*((self.dcr*x/y) + 1)/2)
        r[p] = 1
        return r

    def get_hand_on_retina(self, a):
        if a.sum() != 0:
            dr = (a[:,:] * self.hand_on_retina_filter[:,:]).sum()/a.sum()
            if abs(dr) > 1:
                raise Exception("Hand out of sight.")
            return torch.from_numpy(self.point_on_retina(dr, self.dcr)).float()
        else:
            raise Exception("Couldn't get hand on retina.")

    def abstract_to_concrete_retina(self, abstract_retina):
        return torch.from_numpy(np.array(list(map(lambda x: self.point_on_retina(x, self.D), abstract_retina[:,1]))).sum(axis=0)).float()

    def update_retina_to_action_mapper(self):
        self.f = (lambda dr: self.mc.decode(dr))

'''# Jacobien action

def f(self, action, Retina, Previous_Retina):
    dr = Retina[self.wanted_obj, 1] - Previous_Retina[self.wanted_obj, 1]
    return np.array([(dr/self.D)*sin(action[0])**2, (dr/self.D)*sin(action[1])**2])'''