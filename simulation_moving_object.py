from typing import Tuple
import numpy as np
import cv2 as cv
from classes.retinaAndArm import RetinaAndArm
import json
from random import sample
from classes.cargiver import CareGiver
from math import pi, atan, tan
import torch
import torch.nn as nn
from utils import left, right

### Configuration:

with open("env_config.json", "rb") as file:
    env_config = json.load(file)

M = env_config["M"]
N_itt = env_config["N_itt"]
N_exp = env_config["N_exp"]
N_pitt = env_config["N_pitt"]
batch_size = env_config["batch_size"]

nu = env_config["nu"]
epsilon = env_config["epsilon"]

N_objs = env_config['N_objs']

N_anodes = env_config["N_anodes"]
N_snodes = env_config["N_snodes"]

dcr = env_config["dcr"]
des = env_config["des"]
dx = env_config["dx"]
D = env_config["D"]

im_height = env_config["im_height"]
im_width = env_config["im_width"]
margin = env_config["margin"]
line_level = env_config["line_level"]
world_im_name = env_config["world_im_name"]
fps = env_config["fps"]
omega = env_config["omega"]
radius = env_config["radius"]
l1 = env_config["l1"]
l2 = env_config["l2"]
shoulder = [im_width//2, D+line_level]
color = tuple(env_config["color"])
thickness = env_config["thickness"]

### Setting objects:

objs = list(sample([(i,j) for i in range(margin, im_width-margin) for j in range(line_level//4, line_level)], N_objs))
theta_d = [atan((shoulder[1]-obj[1])/(obj[0]-shoulder[0]))+pi*int(obj[0]<=shoulder[0]) for obj in objs]
delta_theta_d = [(right(theta_d[idx], theta_d, epsilon), left(theta_d[idx], theta_d, epsilon)) for idx in range(len(theta_d))]
env_config['theta_d'] = theta_d
env_config['delta_theta_d'] = delta_theta_d

### Simulation class and methods:

def draw_world(objs, T, world) -> None:

    # cv.line(world, (0, line_level), (im_width, line_level), [255, 255, 255], 1)

    for idx, obj in enumerate(objs):
        cv.line(world, (int(shoulder[0]+(shoulder[1]-obj[1])/tan(theta_d[idx]-T*delta_theta_d[idx][0])), obj[1]), (int(shoulder[0]+(shoulder[1]-obj[1])/tan(theta_d[idx]+T*delta_theta_d[idx][1])), obj[1]), [0, 0, 255], 1)
        cv.circle(world, obj, radius, [0, 255, 0], -1)

class Robot(RetinaAndArm):

    q1 = pi/2
    q2 = 0

    def __init__(self, env_config) -> None:

        super().__init__(env_config)

    def draw(self, world) -> None:

        x1 = (int(shoulder[0]+l1*np.cos(self.q1)), int(shoulder[1]-l1*np.sin(self.q1)))
        x2 = (int(x1[0]+l2*np.cos(self.q1+self.q2)), int(x1[1]-l2*np.sin(self.q1+self.q2)))
        cv.line(world,(shoulder[0], shoulder[1]),(x1[0],x1[1]),color,thickness)
        cv.line(world,(x1[0],x1[1]),(x2[0],x2[1]),color,thickness)
        cv.circle(world, (shoulder[0], shoulder[1]), radius//2, color, -1)
        cv.circle(world, (x1[0], x1[1]), radius//2, color, -1)

    def go_to(self, action: Tuple[float], T: float) -> None:

        new_q1 = action[0]
        new_q2 = action[1]
        delta_q1 = new_q1 - self.q1
        delta_q2 = new_q2 - self.q2
        dt = 1.0/fps 

        if delta_q1 != 0:
            omega1 = omega*(delta_q1/abs(delta_q1)) 
            omega2 = omega*(delta_q2/abs(delta_q1)) 
            N_frames = fps*abs(delta_q1)/omega
        elif delta_q1 == 0 and delta_q2 != 0:
            omega1 = 0
            omega2 = omega*(delta_q2/abs(delta_q2)) 
            N_frames = fps*abs(delta_q2)/omega
        else:
            return

        for _ in range(int(N_frames)):
            world = np.zeros((im_height,im_width,3), np.uint8)
            self.q1 += omega1*dt
            self.q2 += omega2*dt
            self.draw(world)
            draw_world(objs, T, world)
            cv.imshow(world_im_name, world)
            cv.waitKey(int(1000*dt)) 

        world = np.zeros((im_height,im_width,3), np.uint8)
        self.q1 += (N_frames-int(N_frames))*omega1*dt
        self.q2 += (N_frames-int(N_frames))*omega2*dt
        self.draw(world)
        draw_world(objs, T, world)
        cv.imshow(world_im_name, world)
        cv.waitKey(int(1000*dt))

### Setting agents:

env_config["nu"] = 0.3
cg = CareGiver(env_config)
robot = Robot(env_config)
mse = nn.MSELoss()

### Experiment and simulation:

# Phase one:

print("Phase one started")

running_loss = 0.0

for pitt in range(N_pitt):

    a = np.zeros((N_anodes, N_anodes))
    a[tuple(np.array(sample(range(N_anodes), 2)))] = 1
    a = torch.from_numpy(a).float()
    try:
        dr = robot.get_hand_on_retina(a)
    except Exception:
        continue
    a_hat = robot.mc(a)
    dr_hat = robot.mc.encode(a)

    robot.optimizer.zero_grad()
    loss = 0.5*torch.norm(a.view(-1, N_anodes, N_anodes) - a_hat) + 0.5*torch.norm(dr.view(1, -1) - dr_hat)
    loss.backward()
    robot.optimizer.step()

    running_loss += loss.item()
    if pitt % batch_size == 0:
        if pitt == 0:
            loss_0 = running_loss
            print('Pointing learning loss {}/{}: {}'.format(pitt, N_pitt, loss_0))
        else:
            print('Pointing learning loss {}/{}: {}%'.format(pitt, N_pitt, 100*(running_loss/batch_size - loss_0)/loss_0))
            running_loss = 0.0

robot.update_retina_to_action_mapper()

print("Phase one ended")

# Phase two:

print("Phase two started")

for itt in range(N_itt):

    robot.decided_wanted_obj()
    action = robot.take_next_action()
    robot.go_to(action, cg.T)
    R, _ = cg.give_reward(action, robot.wanted_obj)
    robot.update_inner_state(action, R, abstract_retina)

    objs[0] += dx