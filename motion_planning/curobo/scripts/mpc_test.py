import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

## import curobo:

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument("--robot", type=str, default="ur5e_gripper.yml", help="robot configuration to load")
args = parser.parse_args()

###########################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# Third Party
# Enable the layers and stage windows in the UI
# Standard Library
import os

# Third Party
import carb
import numpy as np
from helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper

############################################################


########### OV #################;;;;;


###########
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
########### frame prim #################;;;;;


# Standard Library
from typing import Optional

# Third Party
from helper import add_extensions, add_robot_to_scene

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid, Mesh
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml, get_task_configs_path
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

############################################################


def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)

############################################################
def remap_pose(xyzw_pose):
    # Extract the position and orientation elements based on the remapping
    position = np.array([xyzw_pose[i] for i in [0, 1, 2]])
    orientation = np.array([xyzw_pose[i] for i in [6, 3, 4, 5]])
    # Return as a tuple of numpy arrays
    return (position, orientation)

def remap_pose1(xyzw_pose):
    return [xyzw_pose[i] for i in [0, 1, 2, 6, 3, 4, 5]]


def main():
    # Predefined cfg file with sample poses, robot cfg, and environment cfg
    cfg_file = load_yaml(join_path(get_world_configs_path(), "sample_curobo_config_shelf_to_sink_UPDATE.yaml"))
    robot_file = load_yaml(join_path(get_robot_configs_path(), cfg_file["config_file"]))

    # Custom Poses for Testing
    pose_idx = 0
    predefined_poses = cfg_file["x_desired"]
    target_pose = remap_pose(predefined_poses[pose_idx])
    pose_idx_list =[]     # For debugging purposes    
    failed_pose_idx = []
    
    # Warmup CuRobo instance
    usd_help = UsdHelper()
    tensor_args = TensorDeviceType()

    # Load Robot Configuration
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, cfg_file["config_file"]))["robot_cfg"]
    # Useful robot configuration parameters
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 4}

    # Load Environment Configuration from yaml
    cubes = []
    meshes = []
    env = cfg_file['env']
    for obj in env:
        cube = Cuboid(name=obj['name'], pose=remap_pose1(obj['pose']), dims=obj['dim'])
        cubes.append(cube)
        meshes.append(cube.get_mesh())
        
    world_cfg = WorldConfig(cuboid=cubes, mesh=meshes)
    past_pose = None



    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]

    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    cfg_test = load_yaml(join_path(get_task_configs_path(), "base_cfg_test.yml"))
    # MPC Solver Configuration
    n_obstacle_cuboids = 30     # Caching collision data
    n_obstacle_mesh = 100
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
        base_cfg=cfg_test,
    )

    mpc = MpcSolver(mpc_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )
    
    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=100)

    ########## ISAAC SIM INITIALIZATION ###########
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform) 
    stage.DefinePrim("/curobo", "Xform")
    # Interfacing Isaac Sim with CuRobo
    add_extensions(simulation_app, args.headless_mode)

    usd_help.load_stage(my_world.stage) # loads Isaac Sim world to Curobo environment
                                        # This effectively creates a shared stage between Isaac Sim and Curobo
    # usd_help.add_world_to_stage(world_cfg, base_frame="/World") # Adds collision objects defined in yaml file
    for i in range(len(cubes)):
        usd_help.add_cuboid_to_stage(cubes[i])

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()
    # Make a target to follow
    # target = cuboid.VisualCuboid(
    #     "/World/target",
    #     position=np.array([0.5, 0, 0.5]),
    #     orientation=np.array([0, 1, 0, 0]),
    #     color=np.array([1.0, 0, 0]),
    #     size=0.05,
    # )
    ###############################################
    init_curobo = False

    init_world = False
    cmd_state_full = None
    step = 0
    add_extensions(simulation_app, args.headless_mode)
    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(mpc.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 2:
            my_world.reset()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            # robot._articulation_view.set_max_efforts(
            #     values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            # )

        if not init_curobo:
            init_curobo = True
        step += 1
        step_index = step
        if step_index % 1000 == 0:
            print("Updating world")
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
                reference_prim_path=robot_prim_path,
            )

            # obstacles.add_obstacle(world_cfg.cuboid[0])
            mpc.world_coll_checker.load_collision_model(obstacles)

        # position and orientation of target virtual cube:


        # Extract position and orientation, ensuring they are numeric arrays
        ee_translation_goal = np.array(target_pose[0], dtype=np.float32).flatten()
        ee_orientation_teleop_goal = np.array(target_pose[1], dtype=np.float32).flatten()

        ik_goal = Pose(
            position=tensor_args.to_device(ee_translation_goal),
            quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
        )
        goal_buffer.goal_pose.copy_(ik_goal)
        mpc.enable_pose_cost(enable=True)
        mpc.update_goal(goal_buffer)
        
        # if past_pose is None:
        #     past_pose = cube_position + 1.0

        # if np.linalg.norm(cube_position - past_pose) > 1e-3:
        #     # Set EE teleop goals, use cube for simple non-vr init:
        #     ee_translation_goal = cube_position
        #     ee_orientation_teleop_goal = cube_orientation
        #     ik_goal = Pose(
        #         position=tensor_args.to_device(ee_translation_goal),
        #         quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
        #     )
        #     goal_buffer.goal_pose.copy_(ik_goal)
        #     mpc.update_goal(goal_buffer)
        #     past_pose = cube_position

        # if not changed don't call curobo:

        # get robot current state:
        sim_js = robot.get_joints_state()
        js_names = robot.dof_names
        sim_js_names = robot.dof_names

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        if cmd_state_full is None:
            current_state.copy_(cu_js)
        else:
            current_state_partial = cmd_state_full.get_ordered_joint_state(
                mpc.rollout_fn.joint_names
            )
            current_state.copy_(current_state_partial)
            current_state.joint_names = current_state_partial.joint_names
            # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        common_js_names = []
        current_state.copy_(cu_js)

        mpc_result = mpc.step(current_state, max_attempts=100)
        # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        succ = True  # ik_result.success.item()
        cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in cmd_state_full.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)

        cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        cmd_state_full = cmd_state

        art_action = ArticulationAction(
            cmd_state.position.cpu().numpy(),
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
        # positions_goal = articulation_action.joint_positions
        if step_index % 1000 == 0:
            print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        error = mpc_result.metrics.pose_error.item()

        if succ:
            # set desired joint angles obtained from IK:
            if mpc_result.metrics.feasible.item() and error < 0.005:
                robot.set_joint_positions(default_config, idx_list)
                pose_idx += 1
                target_pose = remap_pose(predefined_poses[pose_idx])
                if not pose_idx >= len(predefined_poses):
                    target_pose = remap_pose(predefined_poses[pose_idx])
                    continue
                else:
                    break
            else:
                for _ in range(3):
                    articulation_controller.apply_action(art_action)

        else:
            carb.log_warn("No action is being taken.")


############################################################

if __name__ == "__main__":
    main()
    simulation_app.close()
