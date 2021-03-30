import numpy as np
from math import sqrt, pi, exp, sin
from copy import deepcopy

### Globals:

class EyeAndArm:

    Previous_Retina = None

    def __init__(self, env_conf):

        self.RANDOM_CHOICE = env_conf['RANDOM_CHOICE']
        self.N_nodes = env_conf['N_nodes']
        self.epsilon = env_conf['epsilon']
        self.h = env_conf['h']
        self.N_objs = env_conf['N_objs']
        self.D = env_conf['D']
        # Initialisation nulle, choix au hasard
        if self.RANDOM_CHOICE:
            self.internal_map = np.zeros((self.N_objs, self.N_nodes, self.N_nodes)) 
        # Initialisation au hasard, choix déterministe
        if not self.RANDOM_CHOICE:
            self.internal_map = np.random.randn(self.N_objs, self.N_nodes, self.N_nodes)
        self.wanted_obj = None
        self.std_e = 8*pi/self.N_nodes 
        self.std_i = 5*pi/self.N_nodes 

    def decided_wanted_obj(self):
        self.wanted_obj = np.random.choice(range(self.N_objs))

    def take_next_action(self):
        # Initialisation nulle, choix au hasard
        if self.RANDOM_CHOICE:
            return tuple(((1-2*self.epsilon)*pi/self.N_nodes)*t+self.epsilon*pi for t in np.unravel_index(np.random.choice(np.flatnonzero(self.internal_map[self.wanted_obj,:,:] == self.internal_map[self.wanted_obj,:,:].max())), (self.N_nodes, self.N_nodes)))
        # Initialisation au hasard, choix déterministe
        if not self.RANDOM_CHOICE:
            return tuple(((1-2*self.epsilon)*pi/self.N_nodes)*t+self.epsilon*pi for t in np.unravel_index(self.internal_map[self.wanted_obj,:,:].argmax(), (self.N_nodes, self.N_nodes)))

    def update_inner_state(self, action, R, Retina):

        if self.Previous_Retina is None:
            self.Previous_Retina =  Retina

        # Building the kernel :
        N = -(((np.linspace(self.epsilon*pi,(1-self.epsilon)*pi,self.N_nodes) - (action[0] + self.f(action, Retina, self.Previous_Retina)[0]))**2).reshape((self.N_nodes, 1)) + (np.linspace(self.epsilon*pi,(1-self.epsilon)*pi,self.N_nodes) - (action[1] + self.f(action, Retina, self.Previous_Retina)[1]))**2)/2
        if R > 0:
            N /= self.std_e**2
            N = np.exp(N)/(self.std_i*sqrt(2*pi))
        else:
            N /= self.std_i**2
            N = np.exp(N)/(self.std_i*sqrt(2*pi))

        # Applying the kernel :
        self.internal_map[self.wanted_obj, :, :] = ((self.h - 1)/self.h)*self.internal_map[self.wanted_obj, :, :] + R*N

    def f(self, action, Retina, Previous_Retina):
        delta_Retinas = Retina[self.wanted_obj, 1] - Previous_Retina[self.wanted_obj, 1]
        return np.array([(delta_Retinas/self.D)*sin(action[0])**2, (delta_Retinas/self.D)*sin(action[1])**2])