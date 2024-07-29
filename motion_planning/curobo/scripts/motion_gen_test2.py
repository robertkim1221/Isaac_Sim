#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Third Party
import torch

# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="ur5e_gripper.yml", help="robot configuration to load")
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=True,
)

args = parser.parse_args()

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.rollout.rollout_base import Goal

from curobo.geom.types import WorldConfig, Cuboid, Cylinder
from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric
)
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

def draw_line(start, gradient):
    # Third Party
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    # if draw.get_num_points() > 0:
    draw.clear_lines()
    start_list = [start]
    end_list = [start + gradient]

    colors = [(1, 0, 0, 0.8)]

    sizes = [10.0]
    draw.draw_lines(start_list, end_list, colors, sizes)

############################################################
def remap_pose(xyzw_pose):
    return [xyzw_pose[i] for i in [0, 1, 2, 6, 3, 4, 5]]
    #[0.31075111653141246, 0.029414976865931558, -0.000580827233999992, 0.6830127018922193, 0.18301270189221935, 0.6830127018922193, -0.18301270189221935]
    #[0.5, -0.234, 0.206, 0.707, 0.00417, -0.707, 0.00406]
    
def main():
    setup_curobo_logger("warn")
    
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
    env = cfg_file['env']
    for obj in env:
        cubes.append(Cuboid(name=obj['name'], pose=remap_pose(obj['pose']), dims=obj['dim']))
    
    world_cfg = WorldConfig(cuboid=cubes)
    
    ######TESTING ADDING OBJECTS TO WORLD CONFIG#######
    # dish_test = Cylinder(
    #     name="dish",
    #     pose=[0.5, -0.234, 0.206, 1, 0, 0, 0],
    #     color=[1.0, 0.0, 0.0, 1.0],
    #     height=0.01,
    #     radius=0.1,
    # )
    
    # world_cfg.add_obstacle(dish_test)
    
    # print(world_cfg.cylinder)
    ###################################################
    
    # MotionGen Parameters
    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 10
    interpolation_dt = 0.05
    n_obstacle_cuboids = 30     # Caching collision data
    n_obstacle_mesh = 100
    
    # Configure MotionGen
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=24,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
        num_ik_seeds= 32,
        js_trajopt_dt=0.15,
        project_pose_to_goal_frame=False
        )
    motion_gen = MotionGen(motion_gen_config)
    print("Curobo warming up...")
    motion_gen.warmup(enable_graph=False, warmup_js_trajopt=True, parallel_finetune=True)
    print("CuRobo is Ready")

    # Configure Motion Planner
    pose_cost_metric = PoseCostMetric(
        hold_partial_pose = True,
        hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 0, 0, 0, 0]),
        reach_partial_pose=True,
        reach_vec_weight=motion_gen.tensor_args.to_device([1,1, 0, 1, 1,1]),
    )

    
    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=5,
        max_attempts=max_attempts,
        enable_finetune_trajopt=True,
        parallel_finetune=True,
        pose_cost_metric=pose_cost_metric,
    )
    
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
    usd_help.add_world_to_stage(world_cfg, base_frame="/World") # Adds collision objects defined in yaml file
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()

    ###############################################

    # Iteration Variables
    cmd_plan = None
    cmd_idx = 0
    i = 0
    spheres = None # spheres for visualizing robot
    # ignore_substring = ["ur5e_robot", "material", "Cube", "curobo"]
    act_distance = 0.05
    config = RobotWorldConfig.load_from_config(
        robot_file,
        world_cfg,
        collision_activation_distance=act_distance,
        collision_checker_type=CollisionCheckerType.MESH,
    )
    model = RobotWorld(config)
    while simulation_app.is_running():
        my_world.step(render=True)  # Step the simulation
        
        if not my_world.is_playing(): # Wait for user to click play
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue
        
        step_index = my_world.current_time_step_index   # Get current time step index

        if step_index < 2: # Reset the simulation and robot to initial state
            my_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            
        if step_index < 20: # Allow the robot/sim to settle
            continue
        
        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t. ur5e_robot")
            obstacles = usd_help.get_obstacles_from_stage(
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/curobo",
                ],
            ).get_collision_check_world()
            
            motion_gen.update_world(obstacles)
            
            carb.log_info("Synced Curobo world from USD stage.")
        
        # Joint states of the robot in the simulation
        sim_js = robot.get_joints_state()
        sim_js_names = robot.dof_names
        assert not np.any(np.isnan(sim_js.positions)), "isaac sim has returned NAN joint position values."

        # Convert to CuRobo JointState
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        
        # motion_gen.attach_objects_to_robot(
        #     cu_js,
        #     ["/World/obstacles/dish"],
        #     sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        #     # world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], tensor_args),
        # )
        
        # Visualizing collision spheres
        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))
                        
        if cmd_plan is None: # Plan to reach target pose if no plan has been generated
            # Define target pose and IK goal
            ee_translation_goal = target_pose[:3]
            ee_orientation_goal = target_pose[3:]
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_goal),
            )
            # Curobo planning
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            succ = result.success.item()

            if succ:
                carb.log_info("Plan converged to a solution.")
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)

                idx_list = [robot.get_dof_index(x) for x in sim_js_names if x in cmd_plan.joint_names]
                # Define command to pass to robot
                cmd_plan = cmd_plan.get_ordered_joint_state([robot.dof_names[i] for i in idx_list])
                cmd_idx = 0
            else:
                carb.log_warn("Plan did not converge to a solution. No action is being taken.")
                carb.log_warn("Attempting to move to next pose.")
                failed_pose_idx.append(pose_idx)
                pose_idx += 1
                if pose_idx >= len(predefined_poses):
                    carb.log_info(f"Finished all predefined poses. Pose indices: {pose_idx_list}")
                    break
                target_pose = remap_pose(predefined_poses[pose_idx])
        else:
            # Pass in command to robot for each iteration
            cmd_state = cmd_plan[cmd_idx]
            
            ########## ISAAC SIM VISUALIZATION ###########
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            ###############################################
        
            last_sphere = sph_list[0][35]

            x_sph = torch.zeros((1, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
            x_sph[..., :3] = tensor_args.to_device(last_sphere.position).view(1, 1, 1, 3)
            x_sph[..., 3] = torch.tensor(last_sphere.radius, device=tensor_args.device, dtype=tensor_args.dtype)

            d, d_vec = model.get_collision_vector(x_sph)
            d = d.view(-1).cpu()
            p = d.item()
            p = max(1, p * 5)
                        
            sph_position = last_sphere.position
            if d.item() != 0.0:
                draw_line(sph_position, d_vec[..., :3].view(3).cpu().numpy())
            else:
                # Third Party
                from omni.isaac.debug_draw import _debug_draw

                draw = _debug_draw.acquire_debug_draw_interface()
                # if draw.get_num_points() > 0:
                draw.clear_lines()

            cmd_idx += 1 # Iterate to next command
            for _ in range(2):
                my_world.step(render=False) # Presumably two sim steps per iteration
            
            if cmd_idx >= len(cmd_plan.position): # Check if all commands have been executed
                robot.set_joint_positions(default_config, idx_list) # Reset to default config for next pose
                
                pose_idx_list.append(pose_idx) # Log successful pose index              
                pose_idx += 1
                if pose_idx >= len(predefined_poses): # Check if all poses have been executed
                    carb.log_warn(f"Failed poses: {failed_pose_idx}")
                    break
                else:
                    # Move to next pose
                    target_pose = remap_pose(predefined_poses[pose_idx])
                    cmd_idx = 0           # Reset command
                    cmd_plan = None

    simulation_app.close()

if __name__ == "__main__":
    main()
