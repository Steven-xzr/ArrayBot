from isaacgym import gymapi, gymutil, gymtorch

import numpy as np
import torch
import os
from tqdm import tqdm


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# parse arguments
args = gymutil.parse_arguments(
    description="play Isaac Gym",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 4, "help": "Number of Environments"},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}],
    headless=True,
    no_graphics=True,
    )

table_file_root = os.path.join(os.path.expanduser("~"), "TableBot", "urdf")
table_file = "tablebot_225.urdf"

object_file_root = os.path.join(os.path.expanduser("~"), "egad", "eval")
object_file = "C0.obj"

asset_options_table = gymapi.AssetOptions()
asset_options_table.fix_base_link = True
# asset_options_table.default_dof_drive_mode = gymapi.DOF_MODE_POS

# asset_options_object = gymapi.AssetOptions()


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

# PhysX does not support URDF with large DOFs
args.physics_engine = gymapi.SIM_FLEX

# illegal memory access with large robot's DOF (e.g. tablebot_169)
sim_params.physx.num_threads = 1
# sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing FleX and CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.distance = 0.0
gym.add_ground(sim, plane_params)

# # create viewer
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     print("*** Failed to create viewer")
#     quit()
#
# # position the camera
# cam_pos = gymapi.Vec3(-0.6, -0.6, 1.5)
# cam_target = gymapi.Vec3(0.6, 0.6, 0)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("Loading asset '%s' from '%s'" % (table_file, table_file_root))
asset_table = gym.load_asset(sim, table_file_root, table_file, asset_options_table)

# print("Loading asset '%s' from '%s'" % (object_file, object_file_root))
# asset_object = gym.load_asset(sim, object_file_root, object_file, asset_options_object)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset_table)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset_table)
dof_props["stiffness"][:].fill(800.0)
dof_props["damping"][:].fill(40.0)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset_table)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset_table, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']
drive_modes = dof_props['driveMode']

# pos_drive_modes = np.array([gymapi.DOF_MODE_POS] * num_dofs, dtype=gymapi.DofDriveMode)
# dof_props['driveMode'] = pos_drive_modes

lower_limit = lower_limits[0]
upper_limit = upper_limits[0]

target_dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
target_dof_states['pos'][:] = (upper_limit - lower_limit) / 2 + lower_limit


# set up the env grid
num_envs = args.num_envs
num_per_row = int(np.sqrt(num_envs))
spacing = 0.3
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)


# cache useful handles
envs = []
table_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.2)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

    table_handle = gym.create_actor(env, asset_table, pose, "actor", i, 1)
    table_handles.append(table_handle)

    # gym.set_actor_scale(env, table_handle, 10)

    props = gym.get_actor_dof_properties(env, table_handle)
    props["driveMode"][:] = gymapi.DOF_MODE_POS
    props["velocity"][:] = 100.0
    props["stiffness"][:] = 1000.0
    props["damping"][:] = 200.0
    props["effort"][:] = 1000.0

    # NOTE: force = posError * stiffness + velError * damping

    gym.set_actor_dof_properties(env, table_handle, props)

    gym.set_actor_dof_states(env, table_handle, target_dof_states, gymapi.STATE_ALL)

    rigid_props = gym.get_actor_rigid_body_properties(env, table_handle)

    # gym.set_actor_dof_properties(env, table_handle, dof_props)
#
# gym.prepare_sim(sim)
#
# # acquire root state tensor descriptor
# _root_tensor = gym.acquire_actor_root_state_tensor(sim)
#
# # wrap it in a PyTorch Tensor and create convenient views
# root_tensor = gymtorch.wrap_tensor(_root_tensor)
# root_positions = root_tensor[:, 0:3]
# root_orientations = root_tensor[:, 3:7]
# root_linvels = root_tensor[:, 7:10]
# root_angvels = root_tensor[:, 10:13]
#
# offsets = torch.tensor([0, 0.5, 0]).repeat(num_envs).reshape(num_envs, 3)
# root_positions += offsets

# total_dof = gym.get_sim_dof_count(sim)
# dof_per_actor_per_env = gym.get_actor_dof_count(envs[0], 0)
# global_index = gym.get_actor_dof_index(envs[0], table_handles[0], 0, gymapi.DOMAIN_SIM)
#
# _dof_states = gym.acquire_dof_state_tensor(sim)
# dof_states = gymtorch.wrap_tensor(_dof_states)
# gym.refresh_dof_state_tensor(sim)
# mid_positions = torch.Tensor([(upper_limit + lower_limit) / 2]).repeat(total_dof)
# dof_states = mid_states
# gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(mid_states))

# gym.set_actor_dof_states(envs[0], table_handles[0], dof_states, gymapi.STATE_ALL)

counter = 0

# while not gym.query_viewer_has_closed(viewer):
# while True:
for _ in tqdm(range(1000)):

    # done = gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(mid_positions))
    # if counter % 1000 == 0:
    #     print("Done: ", done)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    target_dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    target_dof_states['pos'] = np.random.rand(num_dofs) * (upper_limit - lower_limit) + lower_limit

    # current = counter % num_envs
    if counter % 50 == 0:
        for i in range(num_envs):
            target = np.random.rand(num_dofs).astype(np.float32) * (upper_limit - lower_limit) + lower_limit
            gym.set_actor_dof_position_targets(envs[i], table_handles[i], target)

    # # update the viewer
    # gym.step_graphics(sim)
    # gym.draw_viewer(viewer, sim, True)
    #
    # gym.sync_frame_time(sim)

#     if counter % 1000 == 0:
#         print(counter)
    counter += 1
#

print("Done")

# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)
