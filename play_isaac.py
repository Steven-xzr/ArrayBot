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
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
        {"name": "--gui", "action": "store_true", "help": "Open GUI"}],
    headless=True,
    no_graphics=True,
    )


# asset_options_table.default_dof_drive_mode = gymapi.DOF_MODE_POS

# asset_options_object = gymapi.AssetOptions()


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.substeps = 5
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
# args.physics_engine = gymapi.SIM_FLEX
args.physics_engine = gymapi.SIM_PHYSX

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

robot_root = os.path.join(os.path.expanduser("~"), "TableBot", "urdf")
robot_file = "tablebot_row_15.urdf"
asset_options_robot = gymapi.AssetOptions()
asset_options_robot.fix_base_link = True

object_root = os.path.join(os.path.expanduser("~"), "TableBot", "urdf")
object_file = "ball.urdf"


print("Loading asset '%s' from '%s'" % (robot_file, robot_root))
asset_robot = gym.load_asset(sim, robot_root, robot_file, asset_options_robot)

print("Loading asset '%s' from '%s'" % (object_file, object_root))
asset_object = gym.load_asset(sim, object_root, object_file, gymapi.AssetOptions())

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset_robot)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset_robot)
dof_props["stiffness"][:].fill(800.0)
dof_props["damping"][:].fill(40.0)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset_robot)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset_robot, i) for i in range(num_dofs)]

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
robot_handles = []
object_handles = []

robot_side = 15

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # create robot
    for j in range(robot_side):
    # add actor
        pose_robot = gymapi.Transform()
        pose_robot.p = gymapi.Vec3(0.02 * j, 0, 0.2)
        # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        pose_robot.r = gymapi.Quat(0, 0.0, 0.0, 1)

        robot_handle = gym.create_actor(env, asset_robot, pose_robot, "robot" + str(j), i, 1)
        robot_handles.append(robot_handle)

        # NOTE: force = posError * stiffness + velError * damping
        props = gym.get_actor_dof_properties(env, robot_handle)
        props["driveMode"][:] = gymapi.DOF_MODE_POS
        props["velocity"][:] = 100.0
        props["stiffness"][:] = 1000.0
        props["damping"][:] = 200.0
        props["effort"][:] = 1000.0
        gym.set_actor_dof_properties(env, robot_handle, props)
        gym.set_actor_dof_states(env, robot_handle, target_dof_states, gymapi.STATE_ALL)

    # add object
    pose_object = gymapi.Transform()
    pose_object.p = gymapi.Vec3(0.16, 0.16, 0.4)
    pose_object.r = gymapi.Quat(0, 0.0, 0.0, 1)
    object_handle = gym.create_actor(env, asset_object, pose_object, "object", i, 0)
    # gym.set_actor_scale(env, object_handle, 0.1)  # not supported by Flex
    object_handles.append(object_handle)

    # rigid_props = gym.get_actor_rigid_body_properties(env, robot_handle)

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
for _ in tqdm(range(100)):

    # done = gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(mid_positions))
    # if counter % 1000 == 0:
    #     print("Done: ", done)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    target_dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    target_dof_states['pos'] = np.random.rand(num_dofs) * (upper_limit - lower_limit) + lower_limit

    # current = counter % num_envs
    if counter % 2 == 0:
        for i in range(num_envs):
            for j in range(robot_side):
                target = np.random.rand(num_dofs).astype(np.float32) * (upper_limit - lower_limit) + lower_limit
                gym.set_actor_dof_position_targets(envs[i], robot_handles[i * robot_side + j], target)

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
gym.destroy_sim(sim)
