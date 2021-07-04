import numpy as np
from math import sqrt, pi, exp

### Globals:

class Arm:

    def __init__(self, env_config):

        self.RANDOM_CHOICE = env_config['RANDOM_CHOICE']
        self.N_anodes = env_config['N_anodes']
        self.epsilon = env_config['epsilon']
        self.h = env_config['h']
        self.N_objs = env_config['N_objs']
        # Initialisation nulle, choix au hasard
        if self.RANDOM_CHOICE:
            self.internal_map = np.zeros((self.N_objs, self.N_anodes, self.N_anodes)) 
        # Initialisation au hasard, choix déterministe
        if not self.RANDOM_CHOICE:
            self.internal_map = np.random.randn(self.N_objs, self.N_anodes, self.N_anodes)
        self.wanted_obj = None
        self.std_e = 2.0*pi/self.N_anodes 
        self.std_i = 1.5*pi/self.N_anodes

    def decided_wanted_obj(self):
        self.wanted_obj = np.random.choice(range(self.N_objs))

    def take_next_action(self):
        # Initialisation nulle, choix au hasard
        if self.RANDOM_CHOICE:
            return tuple(((1-2*self.epsilon)*pi/self.N_anodes)*t+self.epsilon*pi for t in np.unravel_index(np.random.choice(np.flatnonzero(self.internal_map[self.wanted_obj,:,:] == self.internal_map[self.wanted_obj,:,:].max())), (self.N_anodes, self.N_anodes)))
        # Initialisation au hasard, choix déterministe
        if not self.RANDOM_CHOICE:
            return tuple(((1-2*self.epsilon)*pi/self.N_anodes)*t+self.epsilon*pi for t in np.unravel_index(self.internal_map[self.wanted_obj,:,:].argmax(), (self.N_anodes, self.N_anodes)))

    def update_inner_state(self, action, R, kernel = None):

        if kernel is None or kernel == 'dirac':

            action = np.unravel_index(np.random.choice(np.flatnonzero(self.internal_map[self.wanted_obj,:,:] == self.internal_map[self.wanted_obj,:,:].max())), (self.N_anodes, self.N_anodes))

            # Applying the dirac kernel:
            self.internal_map[self.wanted_obj, action[0], action[1]] = ((self.h - 1)/self.h)*self.internal_map[self.wanted_obj, action[0], action[1]] + R

        elif kernel == 'gaussian':

            # Building the kernel :
            N = -(((np.linspace(self.epsilon*pi,(1-self.epsilon)*pi,self.N_anodes) - action[0])**2).reshape((self.N_anodes, 1)) + (np.linspace(self.epsilon*pi,(1-self.epsilon)*pi,self.N_anodes) - action[1])**2)/2
            if R > 0:
                N /= self.std_e**2
                N = np.exp(N)/(self.std_i*sqrt(2*pi))
            else:
                N /= self.std_i**2
                N = np.exp(N)/(self.std_i*sqrt(2*pi))

            # Applying the kernel :
            self.internal_map[self.wanted_obj, :, :] = ((self.h - 1)/self.h)*self.internal_map[self.wanted_obj, :, :] + R*N

        else:

            raise(Exception('No such kernel exists.'))