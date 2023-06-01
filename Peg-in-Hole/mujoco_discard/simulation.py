import mujoco_py as mjpy
import numpy as np
from gym import spaces
from rotations import *
import utils
import ikpy
import time
import copy

class Simulation:
    def __init__(self, model="model/ur5gripper_new.xml", nsubsteps=2):
        """
        nsubsteps: is equal to the frame skips
        """
        self.model_path=model
        self.urdf_file="model/ur5_gripper.urdf"
        #self.nsubsteps=nsubsteps
        self.model=mjpy.load_model_from_path(self.model_path)
        
        self.sim=mjpy.MjSim(self.model, nsubsteps=nsubsteps)# nsubsteps= self.nsubsteps
        # actuator and joint numbers
        self.arm_joint_names=[name for name in self.model.joint_names if "UR5" in name and "gripper" not in name] # 6
        self.gripper_joint_names=[name for name in self.model.joint_names if "gripper" in name] # 2
        self.body_names=[name for name in self.model.body_names if "UR5" in name or "object" in name]
        self.dict_body={name: self.model.body_name2id(name) for name in self.body_names}
        """
        dict_body contents:
        {'UR5:base_link': 3, 'UR5:shoulder_link': 4, 'UR5:upper_arm_link': 5, 'UR5:forearm_link': 6, 'UR5:wrist_1_link': 7, 
        'UR5:wrist_2_link': 8, 'UR5:wrist_3_link': 9, 'UR5:ee_link': 10, 'UR5:gripper_link': 11, 'UR5:r_gripper_finger_link': 12, 
        'UR5:l_gripper_finger_link': 13, 'object': 14}
        """
        
        self.n_arms=len(self.arm_joint_names)# 6
        self.n_grippers=len(self.gripper_joint_names)#2
        
        self.n_all=len(self.arm_joint_names)+ len(self.gripper_joint_names) # 8 6+2 except the last freejoint
        # set joint pos and vel 
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.qpos=self.sim.data.qpos[:self.n_all]#np.zeros(self.nq)
        self.qvel=self.sim.data.qvel[:self.n_all]#np.zeros(self.nq)
        # action space
        self.n_action=4#7
        self.action_space=spaces.Box(-1.0, 1.0, shape=(self.n_action,), dtype="float32")
               
        # render
        self.fps=1
        self.moving_frame=50
        self.viewer=mjpy.MjViewerBasic( self.sim )
        self.t=0
        self.sim.model.opt.timestep=0.0001#0.002
        self.T=100
        self.n_steps=0
        self.catch_t=10
        self.raise_t=10
        # target position
        #self.target = [0.6, 0.6, 1.7]
        self.target_pos=self.sim.data.get_site_xpos("object")
        self.grip_pos=self.sim.data.get_site_xpos("UR5:finger_site")
        self.ee_chain=ikpy.chain.Chain.from_urdf_file(self.urdf_file)

     
    def init(self):

        self.qpos=np.zeros(self.n_all)
        self.qvel=np.zeros(self.n_all)
        #self.sim.data.body_xpos[-1]=self.target
        # self.current_quat={}
        # for idx, name in enumerate (self.sim.model.body_names):
        #     if "UR5" in name:
        #         self.current_quat.update({name:self.sim.data.get_body_xquat(name)})
        self.sim.forward()

    def step(self, m_action, g_action):
        #action = np.clip(action, self.action_space.low, self.action_space.high)
        #obs=self.get_obs()
        #fix_link=['UR5:wrist_1_link',"UR5:wrist_2_link","UR5:wrist_3_link","UR5:ee_link","UR5:robotiq_85_base_link"]
        while self.n_steps<self.T:      
            if self.n_steps % self.fps==0:
                self.render()     
              
            if self.n_steps<=self.moving_frame:
                self.move(m_action)
            #     self.sim.data.qvel[:self.n_actuator]=0
            #     self.sim.data.qpos[:self.n_actuator]=m_action
            else:
                self.grasp(m_action, g_action)
                # self.sim.data.qvel[:self.n_actuator]=0
                # self.sim.data.qpos[:self.n_actuator]=m_action    
            # self.sim.data.ctrl[:self.n_actuator]=action

            self.sim.step()
            self.n_steps+=1
            self.t=self.sim.data.time

    def get_obs(self):
        pass
    
    def get_quat(self):
        # legacy
        self.quat_dict={}
        for idx, name in enumerate (self.sim.model.body_names):
            if "UR5" in name:
                self.quat_dict.update({name:self.sim.data.get_body_xquat(name)})
        return self.quat_dict

    def move(self, m_action):
        """
        move the arm at first self.T
        """
        self.sim.data.qvel[:self.n_arms]= m_action/self.moving_frame
        self.sim.data.qpos[:self.n_arms]=self.sim.data.qvel[:self.n_arms]*(1+self.n_steps)

        self.sim.data.qvel[self.n_arms:self.n_all]=0#0.2/self.moving_frame
        self.sim.data.qpos[self.n_arms:self.n_all]=0.05#self.sim.data.qvel[self.n_arms:self.n_all]*(1+self.n_steps)
            
    
    def grasp(self, m_action, g_action):
        """
        move down the gripper to the target at the next self.moving_frame-self.catch_t time
        grasp with self.catch_t time

        m_action: moving action
        g_action: grasp action
        """
        down_action=(g_action-m_action)
        up_action=(m_action-g_action)
        l_action=g_action.copy()
        #l_action[-2:]=0.2
        c_action=down_action.copy()
        #c_action[-2:]=0.2
        grasp_t=(self.T-self.moving_frame-self.catch_t)   
        temp=0.05
        if self.n_steps <= self.T - self.catch_t- self.raise_t: # down the gripper
            #self.adjust_orientation()
            self.sim.data.qvel[:self.n_arms]=down_action/grasp_t
            self.sim.data.qpos[:self.n_arms]=self.sim.data.qvel[:self.n_arms]*(self.n_steps-self.moving_frame) + m_action

            self.sim.data.qvel[self.n_arms:self.n_all]=0#
            self.sim.data.qpos[self.n_arms:self.n_all]=temp#-temp/grasp_t*(self.n_steps-self.moving_frame)

        elif self.n_steps <= self.T - self.catch_t: # catch the target     
            catch_t=self.n_steps+ self.catch_t + self.raise_t - self.T
            #temp_t=self.n_steps+ self.catch_t- self.T
            self.sim.data.qvel[:self.n_arms]=(c_action-down_action)/self.catch_t
            self.sim.data.qpos[:self.n_arms]=self.sim.data.qvel[:self.n_arms]*catch_t + g_action

            self.sim.data.qvel[self.n_arms:self.n_all]=-temp/self.catch_t#0
            self.sim.data.qpos[self.n_arms:self.n_all]=-temp/self.catch_t*catch_t#0.02

        else: # lift the gripper
            lift_t= self.n_steps+ self.raise_t - self.T
            self.sim.data.qvel[:self.n_arms]=up_action/self.raise_t
            self.sim.data.qpos[:self.n_arms]=self.sim.data.qvel[:self.n_arms]*(lift_t) +l_action

            self.sim.data.qvel[self.n_arms:self.n_all]=0
            self.sim.data.qpos[self.n_arms:self.n_all]=0.02
        #print("sim g:",self.sim.data.qpos[:self.n_actuator],"g", g_action)

    def stay(self, duration, action):
        """
        Holds the current position by actuating the joints towards their current target position.

        Args:
            duration: Time in ms to hold the position.
        """
        action=np.append(action,np.array([0,0]))
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            self.sim.data.qvel[:self.n_all]=0#[0,0,0,0,0,0,1]
            self.sim.data.qpos[:self.n_all]=action#np.concatenate(action[:self.n_actuator-1],np.array([1*elapsed]))
            elapsed = (time.time() - starting_time)* 1000
    
        #print("stay",self.sim.data.qpos[:self.n_all])

    def finger_ctrl(self, ctrl):
        try:
            assert (len(ctrl)==2),"Invalid control signal to gripper fingers! Specify the list of length 2"
            
            self.sim.data.qpos[-1]=ctrl[0]
            self.sim.data.qpos[-2]=ctrl[1]


        except Exception as e:
            print(e)
            print("Could not control the finger move") 

    def adjust_orientation(self):
        """
        NOT IMPLEMENT YET!!!
        """
        target_idx=self.dict_body["object"]
        grip_link_idx=self.dict_body["UR5:gripper_link"]

        target_quat=self.model.body_quat[target_idx]
        self.model.body_quat[grip_link_idx]=[0.92378524, 0.38291098,0,0]#target_quat
        # return euler

    def ik(self, ee_position):

        try:
            assert (len(ee_position) == 3),"Invalid EE target! Please specify XYZ-coordinates in a list of length 3."

            # to spedify the ee position in world coordinates, so subtract the position of the
            # base link. This is because the inverse kinematics solver chain starts at the base link.
            ee_position_base = ( ee_position - self.sim.data.body_xpos[self.model.body_name2id("UR5:base_link")])
            joint_angles=self.ee_chain.inverse_kinematics(ee_position_base, [0, 0, -1], orientation_mode="X")
            prediction = (self.ee_chain.forward_kinematics(joint_angles)[:3, 3]+
                                            self.sim.data.body_xpos[self.model.body_name2id("UR5:base_link")])
            diff = abs(prediction - ee_position)
            error = np.sqrt(diff.dot(diff))
            joint_angles = joint_angles[1:-1]
            
            # print("error",error)
            # print("pre",prediction ,"ee", ee_position_base)
            
            if error <= 0.02:
                return joint_angles

            print("Failed to find IK solution.")
            return None
        except Exception as e:
            print(e)
            print("Could not find an inverse kinematics solution.")


    def sample_action(self):
        action=np.zeros(self.n_action)
        #for i in range(self.n_action):
        action=np.random.uniform(-1,1,self.n_action)
        #action[-1]=1

        return action

    def render(self):
        self.viewer.render()
        self.sim.forward()

    def reset(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self.n_steps=0

        return True

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}