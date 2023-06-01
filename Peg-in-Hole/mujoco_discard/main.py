#from simulation import Simulation for the 
from simulation_copy import Simulation
import numpy as np
import random
import copy
import time

N_EPISODES = 10
N_STEPS = 100

env=Simulation()

# ini_state=env.sim.get_state()
for epoch in  range (1,N_EPISODES + 1):
    env.init()
    # r1=random.uniform(-1.57,1.57)
    action_m=np.array([0.5,0.1,2,np.pi])#env.grip_pos
    # action_a=copy.deepcopy(action_m)
    action_a=np.array([0.5,0.1,1.8, np.pi])
    # action=env.sample_action()
    m_action=env.ik(action_m[:3]) # arms move action
    g_action=env.ik(action_a[:3]) # gripper grasp action
    
    for step in range(N_STEPS):

        env.step(m_action, g_action)
    env.stay(1000, g_action)
        #print(step,quat['UR5:wrist_1_link'], "euler",env.set_topdown())

    #print(epoch,step,env.t)
    #print(f'quatertion:{env.model.body_quat[11]}, object {env.model.body_quat[14]}')

    print(f"m {m_action}, g {g_action}")
        #print("joint", env.sim.data.ctrl[:])
    env.reset()
env.close()
