from math import sqrt, pi, exp

class CareGiver:

    s = 0

    def __init__(self, env_conf):

        self.nu = env_conf['nu']
        self.T_0 = 10*pi/env_conf['N_nodes']
        self.T_f = 3*pi/env_conf['N_nodes']
        self.T = self.T_0
        self.theta_d = env_conf['theta_d']

    def give_reward(self, action, wanted_obj):

        d = sqrt((action[0]- self.theta_d[wanted_obj])**2+(action[1]- self.theta_d[wanted_obj])**2)

        if d < self.T:
            R = 1
            self.s += 1
            self.T  = (self.T_0 - self.T_f)*exp(-(self.nu*self.s)) + self.T_f
        else:
            R = -1

        return R