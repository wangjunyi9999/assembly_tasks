from simulation_copy import Simulation
import numpy as np
import time
import mujoco_py as mjpy


N_EPISODES = 1
N_STEPS =100000 #100000
MOVE_STEPS=20000#20000
GRASP_STEPS=20000#20000
CATCH_STEPS=40000#40000#40000
STAY_STEPS=5000#5000

fps=100#180 #skip frame

env=Simulation()

for epoch in  range (1,N_EPISODES + 1):
    env.init()

    # action_m=np.array([0.5,0.1,2,np.pi])#env.grip_pos

    # action_a=np.array([0.5,0.1,1.83, np.pi])

    # m_action=env.ik(action_m[:3]) # arms move action
    # g_action=env.ik(action_a[:3]) # gripper grasp action
    """
    m [-1.83346465e-02, -1.44315481e+00,  1.91532038e+00, -2.04296190e+00, -1.57079633e+00,  2.70467080e-09], 
    g [-1.83346411e-02, -1.15676989e+00 , 2.09325725e+00, -2.50728369e+00,-1.57079633e+00,  5.25724221e-09]
    g2[-1.83346443e-02, -1.08903465e+00,  2.10537340e+00, -2.58713508e+00,-1.57079633e+00, -3.59276342e-09]
    """
    n=8
    action_m=np.array([-1.83346465e-02, -1.44315481e+00,  1.91532038e+00, -2.04296190e+00, -1.57079633e+00,  2.70467080e-09, 0.2,0.2])
    #action_g=np.array([-1.83346411e-02, -1.15676989e+00 , 2.09325725e+00, -2.50728369e+00,-1.57079633e+00,  5.25724221e-09,0.2,0.2])
    action_g=np.array([-1.83346443e-02, -1.08903465e+00,  2.10537340e+00, -2.58713508e+00,-1.57079633e+00, -3.59276342e-09,0.2,0.2])
    action_c=np.array([-1.83346411e-02, -1.15676989e+00 , 2.09325725e+00, -2.50728369e+00,-1.57079633e+00,  5.25724221e-09,-0.5,-0.5])
    v_move=action_m/MOVE_STEPS
    v_down=(action_g-action_m)/GRASP_STEPS
    v_catch=(action_c-action_g)/CATCH_STEPS

    for n_steps in range(N_STEPS):
        if n_steps % fps==0:
            env.render()
        if n_steps<=MOVE_STEPS:
            env.sim.data.qvel[:n]=v_move
            env.sim.data.qpos[:n]=v_move*n_steps
        elif MOVE_STEPS<n_steps<=MOVE_STEPS+STAY_STEPS:
            env.sim.data.qvel[:n]=0
            env.sim.data.qpos[:n]=action_m

        elif MOVE_STEPS+STAY_STEPS<n_steps<=MOVE_STEPS+STAY_STEPS+GRASP_STEPS:
            env.sim.data.qvel[:n]=v_down
            env.sim.data.qpos[:n]=v_down*(n_steps-(MOVE_STEPS+STAY_STEPS))+action_m
        elif MOVE_STEPS+STAY_STEPS+GRASP_STEPS<n_steps<=MOVE_STEPS+GRASP_STEPS+STAY_STEPS:
            env.sim.data.qvel[:n]=0
            env.sim.data.qpos[:n]=action_g
            env.model.body_quat[11]=[1, 0, 0, 0]
        elif MOVE_STEPS+GRASP_STEPS+STAY_STEPS<n_steps<=MOVE_STEPS+GRASP_STEPS+STAY_STEPS+CATCH_STEPS:
            env.sim.data.qvel[:n]=v_catch
            env.sim.data.qpos[:n]=v_catch*(n_steps-(MOVE_STEPS+GRASP_STEPS+STAY_STEPS))+action_g
        else:
            env.sim.data.qvel[:n]=0
            env.sim.data.qpos[:n]=action_c
            #print(n_steps,"mujoco time",sim.data.time,"real time",time.time()-now)
        #env.step(m_action, g_action)
    #env.stay(1000, g_action)
        env.sim.step()
        n_steps+=1

    #print(f'quatertion:{env.model.body_quat[11]}, object {env.model.body_quat[14]}')

    # print(f"m {m_action}, g {g_action}")

    env.reset()
env.close()