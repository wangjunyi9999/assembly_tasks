"""Factory: class for peg-hole env.

Inherits base class and abstract environment class. Inherited by peg-hole task classes. Not directly executed.

Configuration defined in FactoryEnvPegHole.yaml. Asset info defined in factory_asset_info_insertion.yaml.

Model file:

round_peg_hole_16mm_tight:
    round_peg_16mm_tight:
        urdf_path: 'factory_round_peg_16mm_tight'
        diameter: 0.015994
        length: 0.050
        density: 8000.0
        friction: 0.5
    round_hole_16mm:
        urdf_path: 'factory_round_hole_16mm'
        diameter: 0.0165
        height: 0.0089916
        density: 8000.0
        friction: 0.5
"""

import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi
from .factory_base import FactoryBase
import tasks.factory_control as fc
# import factory_control as fc
from tasks.factory_schema_class_env import FactoryABCEnv
from tasks.factory_schema_config_env import FactorySchemaConfigEnv


class FactoryEnvPegHole(FactoryBase, FactoryABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_env_yaml_params()

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        config_path = 'task/FactoryEnvPegHole.yaml'  # relative to Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        asset_info_path = '../assets/yaml/factory_asset_info_insertion.yaml'
        self.asset_info_peg_hole = hydra.compose(config_name=asset_info_path)
        self.asset_info_peg_hole = self.asset_info_peg_hole['']['']['']['assets']['yaml']  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        #self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        peg_asset, hole_asset = self._import_env_assets() 
        self._create_actors(lower, upper, num_per_row, franka_asset, peg_asset, hole_asset, table_asset)

    def _import_env_assets(self):
        """Set peg and hole asset options. Import assets."""

        urdf_root = os.path.join(f'/home/jy/junyi/SDU_DTU_master_thesis/Peg-in-Hole/peg_in_hole', 'assets', 'urdf')

        peg_options = gymapi.AssetOptions()
        peg_options.flip_visual_attachments = False
        peg_options.fix_base_link = False
        peg_options.thickness = 0.0  # default = 0.02
        peg_options.armature = 0.0  # default = 0.0
        peg_options.use_physx_armature = True
        peg_options.linear_damping = 0.0  # default = 0.0
        peg_options.max_linear_velocity = 1000.0  # default = 1000.0
        peg_options.angular_damping = 0.0  # default = 0.5
        peg_options.max_angular_velocity = 64.0  # default = 64.0
        peg_options.disable_gravity = False
        peg_options.enable_gyroscopic_forces = True
        peg_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        peg_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            peg_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        hole_options = gymapi.AssetOptions()
        hole_options.flip_visual_attachments = False
        hole_options.fix_base_link = True
        hole_options.thickness = 0.0  # default = 0.02
        hole_options.armature = 0.0  # default = 0.0
        hole_options.use_physx_armature = True
        hole_options.linear_damping = 0.0  # default = 0.0
        hole_options.max_linear_velocity = 1000.0  # default = 1000.0
        hole_options.angular_damping = 0.0  # default = 0.5
        hole_options.max_angular_velocity = 64.0  # default = 64.0
        hole_options.disable_gravity = False
        hole_options.enable_gyroscopic_forces = True
        hole_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        hole_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            hole_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        peg_assets = []
        hole_assets = []
        for subassembly in self.cfg_env.env.desired_subassemblies:
            components = list(self.asset_info_peg_hole[subassembly])
            peg_file = self.asset_info_peg_hole[subassembly][components[0]]['urdf_path'] + '.urdf'
            hole_file = self.asset_info_peg_hole[subassembly][components[1]]['urdf_path'] + '.urdf'
            peg_options.density = self.asset_info_peg_hole[subassembly][components[0]]['density']
            hole_options.density = self.asset_info_peg_hole[subassembly][components[1]]['density']
            peg_asset = self.gym.load_asset(self.sim, urdf_root, peg_file, peg_options)
            hole_asset = self.gym.load_asset(self.sim, urdf_root, hole_file, hole_options)
            peg_assets.append(peg_asset)
            hole_assets.append(hole_asset)

        return peg_assets, hole_assets

    def _create_actors(self, lower, upper, num_per_row, franka_asset, peg_assets, hole_assets, table_asset):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        franka_pose = gymapi.Transform()
        franka_pose.p.x = self.cfg_base.env.franka_depth
        franka_pose.p.y = 0.0
        franka_pose.p.z = 0.0
        franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.franka_handles = []
        self.peg_handles = []
        self.hole_handles = []
        self.table_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.peg_actor_ids_sim = []  # within-sim indices
        self.hole_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        self.peg_heights = []
        self.peg_widths = []
        self.hole_widths = []
        self.hole_heights = []
        # self.hole_shank_lengths = []
        self.thread_pitches = []

        for i in range(self.num_envs):

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i + self.num_envs,
                                                      0, 0)
            else:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i, 0, 0)
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_peg_hole[subassembly])

            peg_pose = gymapi.Transform()
            peg_pose.p.x = 0.0
            peg_pose.p.y = self.cfg_env.env.peg_lateral_offset
            peg_pose.p.z = self.cfg_base.env.table_height
            peg_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            peg_handle = self.gym.create_actor(env_ptr, peg_assets[j], peg_pose, 'peg', i, 0, 0)
            self.peg_actor_ids_sim.append(actor_count)
            actor_count += 1

            peg_height = self.asset_info_peg_hole[subassembly][components[0]]['length']
            hole_height = self.asset_info_peg_hole[subassembly][components[1]]['height']
            peg_width = self.asset_info_peg_hole[subassembly][components[0]]['diameter']
            self.peg_heights.append(peg_height)
            self.hole_heights.append(hole_height)
            self.peg_widths.append(peg_width)

            hole_pose = gymapi.Transform()
            hole_pose.p.x = 0.0
            hole_pose.p.y = 0.0
            hole_pose.p.z = self.cfg_base.env.table_height
            hole_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            hole_handle = self.gym.create_actor(env_ptr, hole_assets[j], hole_pose, 'hole', i, 0, 0)
            self.hole_actor_ids_sim.append(actor_count)
            actor_count += 1

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, 'table', i, 0, 0)
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_link7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand', gymapi.DOMAIN_ACTOR)
            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                  gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_rightfinger',
                                                                   gymapi.DOMAIN_ACTOR)
            self.shape_ids = [link7_id, hand_id, left_finger_id, right_finger_id]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                # print('id',shape_id,'fri',self.cfg_base.env.franka_friction,'self.shape_ids',self.shape_ids)
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)

            peg_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, peg_handle)
            peg_shape_props[0].friction = self.asset_info_peg_hole[subassembly][components[0]]['friction']
            peg_shape_props[0].rolling_friction = 0.0  # default = 0.0
            peg_shape_props[0].torsion_friction = 0.0  # default = 0.0
            peg_shape_props[0].restitution = 0.0  # default = 0.0
            peg_shape_props[0].compliance = 0.0  # default = 0.0
            peg_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, peg_handle, peg_shape_props)

            hole_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hole_handle)
            hole_shape_props[0].friction = self.asset_info_peg_hole[subassembly][components[1]]['friction']
            hole_shape_props[0].rolling_friction = 0.0  # default = 0.0
            hole_shape_props[0].torsion_friction = 0.0  # default = 0.0
            hole_shape_props[0].restitution = 0.0  # default = 0.0
            hole_shape_props[0].compliance = 0.0  # default = 0.0
            hole_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, hole_handle, hole_shape_props)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.peg_handles.append(peg_handle)
            self.hole_handles.append(hole_handle)
            self.table_handles.append(table_handle)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.peg_actor_ids_sim = torch.tensor(self.peg_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.hole_actor_ids_sim = torch.tensor(self.hole_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.peg_actor_id_env = self.gym.find_actor_index(env_ptr, 'peg', gymapi.DOMAIN_ENV)
        self.hole_actor_id_env = self.gym.find_actor_index(env_ptr, 'hole', gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.peg_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, peg_handle, 'peg', gymapi.DOMAIN_ENV)
        self.hole_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, hole_handle, 'hole', gymapi.DOMAIN_ENV)
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                     gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                            gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                             'panda_rightfinger', gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)
        
        self.peg_heights = torch.tensor(self.peg_heights, device=self.device).unsqueeze(-1)
        self.peg_widths = torch.tensor(self.peg_widths, device=self.device).unsqueeze(-1)
        self.hole_heights = torch.tensor(self.hole_heights, device=self.device).unsqueeze(-1)
        """
        # comment previous nut-bolt here
        # For computing body COM pos
        
        self.bolt_head_heights = torch.tensor(self.bolt_head_heights, device=self.device).unsqueeze(-1)

        # For setting initial state
        self.nut_widths_max = torch.tensor(self.nut_widths_max, device=self.device).unsqueeze(-1)
        self.bolt_shank_lengths = torch.tensor(self.bolt_shank_lengths, device=self.device).unsqueeze(-1)

        # For defining success or failure
        self.bolt_widths = torch.tensor(self.bolt_widths, device=self.device).unsqueeze(-1)
        self.thread_pitches = torch.tensor(self.thread_pitches, device=self.device).unsqueeze(-1)
        """

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.peg_pos=self.root_pos[:, self.peg_actor_id_env, 0:3]
        self.peg_quat = self.root_quat[:, self.peg_actor_id_env, 0:4]
        self.peg_linvel = self.root_linvel[:, self.peg_actor_id_env, 0:3]
        self.peg_angvel = self.root_angvel[:, self.peg_actor_id_env, 0:3]

        self.hole_pos = self.root_pos[:, self.hole_actor_id_env, 0:3]
        self.hole_quat = self.root_quat[:, self.hole_actor_id_env, 0:4]
        
        self.peg_force= self.contact_force[:, self.peg_body_id_env, 0:3]
        self.hole_force= self.contact_force[:, self.hole_body_id_env, 0:3]

        self.peg_com_pos = fc.translate_along_local_z(pos=self.peg_pos,
                                                      quat=self.peg_quat,
                                                      offset=self.peg_heights * 0.5, 
                                                      device=self.device)# offset=self.hole_heights + self.peg_heights * 0.5
        self.peg_com_quat = self.peg_quat  # always equal
        self.peg_com_linvel = self.peg_linvel + torch.cross(self.peg_angvel,
                                                            (self.peg_com_pos - self.peg_pos),
                                                            dim=1)
        self.peg_com_angvel = self.peg_angvel  # always equal

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.peg_com_pos = fc.translate_along_local_z(pos=self.peg_pos,
                                                      quat=self.peg_quat,
                                                      offset=self.peg_heights * 0.5, # offset=self.socket_heights + self.plug_heights * 0.5
                                                      device=self.device)

        self.peg_com_linvel = self.peg_linvel + torch.cross(self.peg_angvel,
                                                            (self.peg_com_pos - self.peg_pos),
                                                            dim=1)
