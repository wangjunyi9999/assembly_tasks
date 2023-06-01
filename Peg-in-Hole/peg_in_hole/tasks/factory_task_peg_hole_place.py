"""Factory: Class for peg-hole place task.

Inherits peg-hole environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskPegHolePlace
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

class FactoryTaskPegHolePlace(FactoryEnvPegHole, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

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

        ppo_path = 'train/FactoryTaskPegHolePlacePPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""
 
        
        self.peg_base_pos_local= (self.hole_heights + self.cfg_task.rl.peg_over_hole) * \
                    torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        self.hole_pos_local= self.hole_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale

        self.keypoints_peg = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                             dtype=torch.float32,
                                             device=self.device)
        self.keypoints_hole = torch.zeros_like(self.keypoints_peg, device=self.device)

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # # Compute pos of keypoints on gripper and peg in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_peg[:, idx] = torch_jit_utils.tf_combine(self.peg_quat,
                                                                        self.peg_pos,
                                                                        self.identity_quat,
                                                                        (keypoint_offset + self.peg_base_pos_local))[1]
            self.keypoints_hole[:, idx] = torch_jit_utils.tf_combine(self.hole_quat,
                                                                    self.hole_pos,
                                                                    self.identity_quat,
                                                                    (keypoint_offset + self.hole_pos_local))[1]
        
    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=0.0,
                                            do_scale=True)

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
                       self.peg_pos,
                       self.peg_quat,
                       self.hole_pos,
                       self.hole_quat,]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        return self.obs_buf

    def compute_reward(self):
        """Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        keypoint_reward = -self._get_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
       
        if is_last_step:
            # Check if peg is close enough to hole
            is_peg_close_to_hole = self._check_peg_close_to_hole()
            self.rew_buf[:] += is_peg_close_to_hole * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(is_peg_close_to_hole.float())

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
        """Reset root states of peg and hole."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Randomize root state of peg within gripper
        self.root_pos[env_ids, self.peg_actor_id_env, 0] = 0.0
        self.root_pos[env_ids, self.peg_actor_id_env, 1] = 0.0
        fingertip_midpoint_pos_reset = 0.58781  # self.fingertip_midpoint_pos at reset
        peg_base_pos_local = self.cfg_task.rl.peg_finger_com_offset + self.peg_heights.squeeze(-1)*0.5
        self.root_pos[env_ids, self.peg_actor_id_env, 2] = fingertip_midpoint_pos_reset - peg_base_pos_local

        peg_noise_pos_in_gripper = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        peg_noise_pos_in_gripper = peg_noise_pos_in_gripper @ torch.diag(
            torch.tensor(self.cfg_task.randomize.peg_noise_pos_in_gripper, device=self.device))
        self.root_pos[env_ids, self.peg_actor_id_env, :] += peg_noise_pos_in_gripper[env_ids]

        
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

        self.root_linvel[env_ids, self.hole_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.hole_actor_id_env] = 0.0

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

    def _get_keypoint_offsets(self, num_keypoints):
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at 0.
        e.g. if num_keypoints = 2 :
        tensor([[ 0.0000,  0.0000, -0.5000],
        [ 0.0000,  0.0000,  0.5000]])

        if num_keypoints = 4 :
        tensor([[ 0.0000,  0.0000, -0.5000],
        [ 0.0000,  0.0000, -0.1667],
        [ 0.0000,  0.0000,  0.1667],
        [ 0.0000,  0.0000,  0.5000]])
        """

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _get_keypoint_dist(self):
        """Get keypoint distance."""

        keypoint_dist = torch.sum(torch.norm(self.keypoints_hole - self.keypoints_peg, p=2, dim=-1), dim=-1)
        # print(f'peg{self.keypoints_peg}dist{keypoint_dist}')
        return keypoint_dist

    def _check_peg_close_to_hole(self):
        """Check if peg is close to hole."""

        keypoint_dist = torch.norm(self.keypoints_hole - self.keypoints_peg, p=2, dim=-1)

        is_peg_close_to_hole = torch.where(torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
                                           torch.ones_like(self.progress_buf),
                                           torch.zeros_like(self.progress_buf))
        # Peg Hole tensors self.cfg_task.rl.peg_over_hole
        is_peg_over_hole = torch.where(self.peg_pos[ : , 2] > self.cfg_task.rl.peg_over_hole + self.cfg_base.env.table_height,
                                     torch.ones_like(self.progress_buf),
                                     torch.zeros_like(self.progress_buf))

        is_peg_close_to_hole=torch.where(torch.logical_and(is_peg_close_to_hole, is_peg_over_hole),
                                        torch.ones_like(self.progress_buf),
                                        torch.zeros_like(self.progress_buf)) 
        return is_peg_close_to_hole

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
