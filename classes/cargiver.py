from math import sqrt, pi, exp

class CareGiver:

    s = 0

    def __init__(self, env_config):

        self.nu = env_config['nu']
        self.mu = env_config['mu']
        self.w = env_config['w']
        self.T_0 = 0.5
        self.T_f = 0.25
        self.T = self.T_0
        self.alpha = 1
        self.theta_d = env_config['theta_d']
        self.delta_theda_d = env_config['delta_theta_d']
        self.COUNT_ERRORS_WHEN_SCAFFOLDING = env_config['COUNT_ERRORS_WHEN_SCAFFOLDING']

    def give_reward(self, action, wanted_obj):

        d_min = -(action[0]- self.theta_d[wanted_obj])+self.w*abs(action[1])
        d_max = (action[0]- self.theta_d[wanted_obj])+self.w*abs(action[1])
        d = max(d_min, d_max)

        if d_min < self.T*self.delta_theda_d[wanted_obj][0] and d_max < self.T*self.delta_theda_d[wanted_obj][1]:
            R = 1
        else:
            R = -1

        '''if self.COUNT_ERRORS_WHEN_SCAFFOLDING:
            self.s += R
        else:
            self.s += max(0,R)

        self.T  = exp(-self.nu*self.s) * self.T_0 + (1 - exp(-self.nu*self.s)) * self.T_f'''

        if self.COUNT_ERRORS_WHEN_SCAFFOLDING:
            self.s += R
        else:
            self.s += max(0,R)
        
        if self.s >= 0:
            self.T  = (2/(1+exp(self.nu*self.s))) * self.T_0 + (1 - (2/(1+exp(self.nu*self.s)))) * self.T_f
        else:
            self.T  = (2/(1+exp(self.mu*self.s))) * self.T_0 + (1 - (2/(1+exp(self.mu*self.s)))) * self.T_f

        '''if R >= 0:
            self.alpha *= exp(-self.nu)
        else:
            self.alpha = 1 - exp(-(1-self.mu))*(1 - self.alpha)
        
        self.T  = self.alpha * self.T_0 + (1 - self.alpha) * self.T_f'''

        return R, d