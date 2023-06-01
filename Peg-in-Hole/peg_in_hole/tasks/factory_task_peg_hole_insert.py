"""Factory: Class for peg-hole insert task.

Inherits peg-hole environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskPegHoleInsert
"""

import hydra
import omegaconf
import os
import torch

from isaacgym import gymapi, gymtorch, torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_env_peg_hole import FactoryEnvPegHole
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgymenvs.utils import torch_jit_utils
import math

class FactoryTaskPegHoleInsert(FactoryEnvPegHole, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.viewer is not None:
            self._set_viewer_params()

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_peg_hole = hydra.compose(config_name=asset_info_path)
        self.asset_info_peg_hole = self.asset_info_peg_hole['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

        ppo_path = 'train/FactoryTaskPegHoleInsertPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        target_heights= self.cfg_base.env.table_height +  self.cfg_task.rl.insertion_depth_thresh
        self.target_pos = target_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        
        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale

        self.keypoints_peg = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                             dtype=torch.float32,
                                             device=self.device)
        self.keypoints_hole = torch.zeros_like(self.keypoints_peg, device=self.device)
        self.keypoints_finger_midpoint = torch.zeros_like(self.keypoints_peg, device=self.device)

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        # distance between peg COM and target
        self.peg_dist_to_target = torch.norm(self.target_pos - self.peg_com_pos, p=2, dim=-1)

        self.fingerpad_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.hand_quat,
                                                                 offset=self.asset_info_franka_table.franka_finger_length - self.asset_info_franka_table.franka_fingerpad_length * 0.5,
                                                                 device=self.device)

        # distance between nut COM and midpoint between centers of fingerpads
        self.peg_dist_to_fingerpads = torch.norm(self.fingerpad_midpoint_pos - self.peg_com_pos, p=2, dim=-1)   

        # Compute pos of keypoints on gripper and peg in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_finger_midpoint[:, idx] = torch_jit_utils.tf_combine(self.fingertip_midpoint_quat,
                                                                        self.fingertip_midpoint_pos,
                                                                        self.identity_quat,
                                                                        keypoint_offset.repeat(self.num_envs, 1))[1]

            self.keypoints_peg[:, idx] = torch_jit_utils.tf_combine(self.peg_quat,
                                                                    self.peg_pos,
                                                                    self.identity_quat,
                                                                    keypoint_offset.repeat(self.num_envs, 1))[1] 

            self.keypoints_hole[:, idx] = torch_jit_utils.tf_combine(self.hole_quat,
                                                                    self.hole_pos,
                                                                    self.identity_quat,
                                                                    keypoint_offset.repeat(self.num_envs, 1))[1]                                                   

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""
        # print(f'action,{actions}')
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        
        open_step=50
        is_open_step = (self.progress_buf[0] >= self.max_episode_length - open_step)
        # open gripper
        # if is_open_step:
        #     self._apply_actions_as_ctrl_targets(actions=self.actions,
        #                                     ctrl_target_gripper_dof_pos=0.1,
        #                                     do_scale=True)
        # else:
        
        # self._apply_actions_as_ctrl_targets_amp(actions=self.actions,
        #                                     ctrl_target_gripper_dof_pos=0.0,
        #                                     do_scale=True) # Admittance control
        self._apply_actions_as_ctrl_targets( actions=self.actions,
                                            ctrl_target_gripper_dof_pos=0.0,
                                            do_scale=True) # Impedance Control
    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.peg_com_pos,
                       self.peg_com_quat,
                       self.peg_com_linvel,
                       self.peg_com_angvel,
                       self.left_finger_force,
                       self.right_finger_force,]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        return self.obs_buf

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        # Get successful and failed envs at current timestep
        curr_successes = self._get_curr_success()
        curr_failures = self._get_curr_failure(curr_successes)

        self._update_reset_buf(curr_successes, curr_failures)
        self._update_rew_buf(curr_successes)

    def _update_reset_buf(self, curr_successes, curr_failures):
        """Assign environments for reset if successful or failed."""

        #print(f'curr_successes, {curr_successes},curr_failures,{curr_failures}')
        # self.reset_buf[:] = torch.logical_or(curr_successes, curr_failures)
        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""

        keypoint_reward= -self._get_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale \
                            + curr_successes * self.cfg_task.rl.success_bonus

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        
        if is_last_step: 
            is_inserted= self._check_peg_insert_hole()
            self.extras['successes'] = torch.mean(is_inserted.float())

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

             
        # Close gripper onto peg
        self.disable_gravity()  # to prevent peg from falling
        for _ in range(self.cfg_task.env.num_gripper_close_sim_steps):
            self.ctrl_target_dof_pos[env_ids, 7:9] = 0.0
            delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                          device=self.device)  # no arm motion
            self._apply_actions_as_ctrl_targets(actions=delta_hand_pose,
                                                ctrl_target_gripper_dof_pos=0.0,
                                                do_scale=False)
            self.gym.simulate(self.sim)
            self.render()
        self.enable_gravity(gravity_mag=abs(self.cfg_base.sim.gravity[2]))

        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)
    

        self._reset_buffers(env_ids)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        self.dof_pos[env_ids] = torch.cat((torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, 
                                device=self.device).repeat((len(env_ids), 1)),
                                (self.peg_widths * 0.5) * 1.1,  # buffer on gripper DOF pos to prevent initial contact
                                (self.peg_widths * 0.5) * 1.1),  # buffer on gripper DOF pos to prevent initial contact
                                dim=-1)  # shape = (num_envs, num_dofs)

        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
   
    def _reset_object(self, env_ids):
        """Reset root states of peg and hole. Now it fixes not random"""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        """        
        peg_base_pos_local = self.hole_heights.squeeze(-1)

        if self.cfg_task.randomize.initial_state == 'random': # X [-1,1]*0.05 Y [-1, 1]*0.05 Z table+0.05
            self.root_pos[env_ids, self.peg_actor_id_env] = \
            torch.cat((
                        (torch.rand((self.num_envs, 1), device=self.device) * 2.0 - 1.0) * self.cfg_task.randomize.peg_noise_xy,
                        self.cfg_task.randomize.peg_bias_y + (torch.rand((self.num_envs, 1), device=self.device) * 2.0 - 1.0) * self.cfg_task.randomize.peg_noise_xy,
                        torch.ones((self.num_envs, 1), device=self.device) * (self.cfg_base.env.table_height + self.cfg_task.randomize.peg_bias_z)
                        ), dim=1)
            self.root_pos[env_ids, self.peg_actor_id_env, 2] = self.cfg_base.env.table_height+peg_base_pos_local+ self.cfg_task.randomize.peg_bias_z
        elif self.cfg_task.randomize.initial_state == 'goal':
        """
        
        self.root_pos[env_ids, self.peg_actor_id_env] = torch.tensor([0.0, 0.0, self.cfg_base.env.table_height+ self.cfg_task.rl.peg_over_hole], 
                                                                          device=self.device)

        peg_rot_euler = torch.tensor([0.0, 0.0, math.pi * 0.5], device=self.device).repeat(len(env_ids), 1)
        peg_noise_rot_in_gripper = 2 * (torch.rand(self.num_envs, dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        peg_noise_rot_in_gripper *= self.cfg_task.randomize.peg_noise_rot_in_gripper
        peg_rot_euler[:, 2] += peg_noise_rot_in_gripper
        peg_rot_quat = torch_utils.quat_from_euler_xyz(peg_rot_euler[:, 0], peg_rot_euler[:, 1], peg_rot_euler[:, 2])
        self.root_quat[env_ids, self.peg_actor_id_env] = peg_rot_quat

        # Randomize root state of hole
        hole_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        hole_noise_xy = hole_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.hole_pos_xy_noise, dtype=torch.float32, device=self.device))
        self.root_pos[env_ids, self.hole_actor_id_env, 0] = self.cfg_task.randomize.hole_pos_xy_initial[0] + \
                                                            hole_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.hole_actor_id_env, 1] = self.cfg_task.randomize.hole_pos_xy_initial[1] + \
                                                            hole_noise_xy[env_ids, 1]
        self.root_pos[env_ids, self.hole_actor_id_env, 2] = self.cfg_base.env.table_height
        self.root_quat[env_ids, self.hole_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                       device=self.device).repeat(len(env_ids), 1)
                                        

        self.root_linvel[env_ids, self.peg_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.peg_actor_id_env] = 0.0

        peg_hole_actor_ids_sim = torch.cat((self.peg_actor_ids_sim[env_ids],
                                            self.hole_actor_ids_sim[env_ids]),
                                           dim=0)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(peg_hole_actor_ids_sim),
                                                     len(peg_hole_actor_ids_sim))

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        # actions[:,0:2]= self.target_pos[:,0:2]
        # actions[:,2]=self.target_pos[:,2]+0.03
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        
        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _apply_actions_as_ctrl_targets_amp(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        #self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions # impedance
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor([0.0, 0.0, 0.43], device=self.device).repeat(self.num_envs,1)#self.fingertip_midpoint_pos + self.hole_pos[:,0:3]
        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _get_keypoint_dist(self):
        """Get keypoint distance."""

        peg_hole_keypoint_dist = torch.sum(torch.norm(self.keypoints_hole - self.keypoints_peg, p=2, dim=-1), dim=-1)
        finger_peg_keypoint_dist = torch.sum(torch.norm(self.keypoints_finger_midpoint- self.keypoints_peg, p=2, dim=-1), dim=-1)
        
        keypoint_dist = peg_hole_keypoint_dist + finger_peg_keypoint_dist

        return keypoint_dist

    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.1, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                      device=self.device)  # No arm motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self.gym.simulate(self.sim)

    def _check_peg_insert_hole(self):
        """Check if peg is inserted to hole."""

        # keypoint_dist = torch.norm(self.keypoints_hole - self.keypoints_peg, p=2, dim=-1)
        insertion_depth= torch.norm(self.peg_pos-self.hole_pos, p=2, dim=-1)
        is_peg_insert_hole = torch.where(insertion_depth < self.cfg_task.rl.insertion_depth_thresh,
                                           torch.ones_like(self.progress_buf),
                                           torch.zeros_like(self.progress_buf))
        
        #print(f'peg,{self.peg_pos},insert_d,{insertion_depth}')
        return is_peg_insert_hole
   
    def _get_curr_success(self):
        """Get success mask at current timestep."""

        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)        
        is_inserted=self._check_peg_insert_hole()
        curr_successes = torch.logical_or(curr_successes, is_inserted)
       
        return curr_successes

    def _get_curr_failure(self, curr_success):
        """Get failure mask at current timestep.

        record function input and output type
            torch.logical_or(tensor_a, tensor_b): 
                input must be Tensor type 
                return bool type True/False Tensor type
        """

        curr_failures = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # If max episode length has been reached
        self.is_expired = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length,
                                      torch.ones_like(curr_failures),
                                      curr_failures)
        # If peg is far from hole
        self.is_far = torch.where(self.peg_dist_to_target > self.cfg_task.rl.far_thresh,
                                  torch.ones_like(curr_failures),
                                  curr_failures)          
        # if peg slipped case now set slipe offset is 0.5 * franka_fingerpad_length: 0.5 * 0.017608
        self.is_slipped = torch.where(self.peg_dist_to_fingerpads - self.cfg_task.rl.peg_finger_com_offset > self.asset_info_franka_table.franka_fingerpad_length*0.5,
                                    torch.ones_like(curr_failures),
                                    curr_failures)
        self.is_slipped = torch.logical_and(self.is_slipped, torch.logical_not(curr_success))  # ignore slip if successful
        # If peg fell from gripper
        self.is_fallen = torch.where(self.peg_com_pos[:, 2]< self.cfg_base.env.table_height + self.peg_heights.squeeze(
                                    -1) * 0.5,
                                    torch.ones_like(curr_failures),
                                    curr_failures)

        curr_failures = torch.logical_or(curr_failures, self.is_expired)
        curr_failures = torch.logical_or(curr_failures, self.is_far)
        curr_failures = torch.logical_or(curr_failures, self.is_slipped)
        curr_failures = torch.logical_or(curr_failures, self.is_fallen)
        # print(f'expired,{self.is_expired},far,{self.is_far},slip,{self.is_slipped},fall,{self.is_fallen},\
        #         slip_d,{self.peg_dist_to_fingerpads},mid_pos,{self.fingerpad_midpoint_pos}, peg_com,{self.peg_com_pos}')
        return curr_failures


    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self.device)+\
                                        torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_pos_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = \
            fingertip_midpoint_pos_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise,
                                                                   device=self.device))
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                            device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_rot_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2])

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose
          
            self._apply_actions_as_ctrl_targets(actions=actions,
                                                ctrl_target_gripper_dof_pos=0.0,
                                                do_scale=False)

            self.gym.simulate(self.sim)
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
