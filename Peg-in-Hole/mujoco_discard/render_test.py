import mujoco_py as mjpy
import numpy as np
from gym import spaces
from rotations import *
import utils
import ikpy

model_path='model/ur5gripper.xml'
nsubsteps=1  # wait to set
model=mjpy.load_model_from_path(model_path)
sim=mjpy.MjSim(model, nsubsteps=nsubsteps)
#viewer=mjpy.MjViewerBasic(sim) # wait for rendering def
data=sim.data
t=sim.data.time
n_actuator=len(model.actuator_names)# 7
nq=len(model.joint_names)-1 # 9 8+1 freejoint so now nq=8 
#n_geom=len(model.geom_names)
n_actions=7
action_space=spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
has_object=True
target_range=0.1
distance_threshold=0.01
ini_gripper_xpos=sim.data.get_site_xpos("UR5:finger_site").copy()
height_offset = sim.data.get_site_xpos("object")[2]
target_offset=[0.6, 0, 1.7]
viewer=mjpy.MjViewerBasic(sim)
DEFAULT_SIZE=500
action=np.zeros(n_actions)


def sample_goal():
    if has_object:
        goal=ini_gripper_xpos[:3]+ np.random.uniform(-target_range, target_range, size=3)
        goal+=target_offset
        goal[2]=height_offset
    else:
        goal=ini_gripper_xpos[:3]+ np.random.uniform(-target_range, target_range, size=3)
    return goal.copy()

def get_obs(sim):

    grip_pos = sim.data.get_site_xpos("UR5:finger_site")# the mid point position of gripper
    
    grip_velp=sim.data.get_site_xvelp("UR5:finger_site")*dt
    robot_qpos, robot_qvel = utils.robot_get_obs(sim)

    if has_object:
        object_pos=sim.data.get_site_xpos("object")
        # rotations
        object_rot = mat2euler(sim.data.get_site_xmat("object"))
        # velocities
        object_velp = sim.data.get_site_xvelp("object") * dt
        object_velr = sim.data.get_site_xvelr("object") * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
    else:
        object_pos = (
            object_rot
        ) = object_velp = object_velr = object_rel_pos = np.zeros(0)

    gripper_state = robot_qpos[-2:]
    gripper_vel = (
        robot_qvel[-2:] * dt
    )  # change to a scalar if the gripper is made symmetric

    if not has_object:
        achieved_goal = grip_pos.copy()
    else:
        achieved_goal = np.squeeze(object_pos.copy())
    obs = np.concatenate(
        [
            grip_pos,
            object_pos.ravel(),
            object_rel_pos.ravel(),
            gripper_state,
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
            grip_velp,
            gripper_vel,
        ]
    )
    goal= sample_goal()
    return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": goal.copy(),
        }

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def compute_reward(achieved_goal, goal):
    d = goal_distance(achieved_goal, goal)
    return -d
    
def is_success(achieved_goal, desired_goal):
    d = goal_distance(achieved_goal, desired_goal)
    return (d < distance_threshold).astype(np.float32)

def viewer_setup():
    # wait for modify
    body_id = sim.model.body_name2id("UR5:ee_link")
    lookat = sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        viewer.cam.lookat[idx] = value
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 132.0
    viewer.cam.elevation = -14.0

def get_viewer(mode):
    if mode=="human":
        viewer = mjpy.MjViewer(sim)
    elif mode == "rgb_array":
        viewer== mjpy.MjRenderContextOffscreen(sim, device_id=-1)
    viewer_setup()

    return viewer

def render_callback():
    goal= sample_goal()
    sites_offset=sim.data.site_xpos-sim.model.site_pos
    site_id=sim.model.site_name2id("UR5:finger_site")
    sim.model.site_pos[site_id] = goal - sites_offset[0]
    sim.forward()

def render(mode, width=DEFAULT_SIZE, height=DEFAULT_SIZE):
    render_callback()
    if mode == "rgb_array":
        get_viewer(mode).render(width, height)
        # window size used for old mujoco-py:
        data = get_viewer(mode).read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return data[::-1, :, :]
    elif mode == "human":
        get_viewer(mode).render()

def env_setup():
    for name in model.joint_names:
        if "UR5" in name:
            sim.data.set_joint_qpos(name, sim.data.get_joint_qpos(name))
            #print(name,sim.data.get_joint_qpos(name))
        sim.forward()
        

def init(qpos,qvel):
    sim.data.qpos[:nq]=qpos
    sim.data.qvel[:nq]=qvel
    sim.forward()

def step(action):
    action = np.clip(action, action_space.low, action_space.high)
    obs=get_obs(sim)
    
    sim.step()

def reset():
    sim.reset()

def close():
    pass

qpos=[1,0.5,0.2,0,0,0,0,0]
qvel=np.zeros(nq)
#env_setup()
sim.model.opt.timestep=0.0001#0.0001

dt = nsubsteps * sim.model.opt.timestep # 0.005=50 * 0.0001
fps=60#1.0/dt
n_steps=0
init(qpos,qvel)
# from datetime import datetime
import time
tt=0
now=time.time()
#for epoch in range (5):
while t< 300:
    step(action)
    if n_steps % fps==0:
        viewer.render()
        
        #print("n_steps",n_steps)
#get_viewer('human')

    sim.data.body_xpos[-1]=target_offset
    #ini_gripper_xpos = sim.data.get_site_xpos("UR5:finger_site").copy()
    n_steps+=1
    t=data.time
    tt=time.time()-now
    print("mujoco time",t,"real time",time.time()-now)
#print("real time",time.time()-now)
reset()



