import numpy as np
from math import sqrt, pi, exp

### Globals:

class VCortexAndArm:

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
            self.internal_map = np.zeros((self.N_objs, self.N_rnodes, self.N_anodes, self.N_anodes)) 

        # Initialisation au hasard, choix déterministe
        if not self.RANDOM_CHOICE:
            self.internal_map = np.random.randn(self.N_objs, self.N_rnodes, self.N_anodes, self.N_anodes)

        self.wanted_obj = None
        self.std_R_e = pi/self.N_anodes 
        self.std_R_i = 0.72*pi/self.N_anodes 
        self.std_O_e = 1/self.N_rnodes
        self.std_O_i = 0.72/self.N_rnodes

    def decided_wanted_obj(self):

        self.wanted_obj = np.random.choice(range(self.N_objs))

    def point_on_retina(self, x, y):

        p = int(self.N_rnodes*((self.dcr*x/y) + 1)/2)
        return p

    def take_next_action(self, abstract_retina):

        wanted_obj_on_retina = self.point_on_retina(abstract_retina[self.wanted_obj, 1], self.D)

        # Initialisation nulle, choix au hasard
        if self.RANDOM_CHOICE:
            return tuple(((1-2*self.epsilon)*pi/self.N_anodes)*t+self.epsilon*pi for t in np.unravel_index(np.random.choice(np.flatnonzero(self.internal_map[self.wanted_obj,wanted_obj_on_retina,:,:] == self.internal_map[self.wanted_obj,wanted_obj_on_retina,:,:].max())), (self.N_anodes, self.N_anodes)))
        # Initialisation au hasard, choix déterministe
        if not self.RANDOM_CHOICE:
            return tuple(((1-2*self.epsilon)*pi/self.N_anodes)*t+self.epsilon*pi for t in np.unravel_index(self.internal_map[self.wanted_obj,wanted_obj_on_retina,:,:].argmax(), (self.N_anodes, self.N_anodes)))

    def update_inner_state(self, action, R, abstract_retina):

        wanted_obj_on_retina = self.point_on_retina(abstract_retina[self.wanted_obj, 1], self.D)

        if R > 0:
            std_R = self.std_R_e
            std_O = self.std_O_e
        else:
            std_R = self.std_R_i
            std_O = self.std_O_i

        # Building the kernel :
        N = -(((np.linspace(self.epsilon*pi,(1-self.epsilon)*pi,self.N_anodes) - action[0])**2/std_R**2).reshape((1, self.N_anodes, 1)) + ((np.linspace(self.epsilon*pi,(1-self.epsilon)*pi,self.N_anodes) - action[1])**2/std_R**2).reshape(1, 1, self.N_anodes) + ((np.linspace(-1, 1, self.N_rnodes) - wanted_obj_on_retina)**2/std_O**2).reshape(self.N_rnodes, 1, 1))/2
        N = np.exp(N)/(std_R**2*std_O*sqrt((2*pi)**3))

        # Applying the kernel :
        self.internal_map[self.wanted_obj, :, :, :] = (1 - 1/self.h)*self.internal_map[self.wanted_obj, :, :, :] + (1/self.h)*R*N