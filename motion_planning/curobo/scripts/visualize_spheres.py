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

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
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

############################################################
def remap_pose(xyzw_pose):
    return [xyzw_pose[i] for i in [0, 1, 2, 6, 3, 4, 5]]
    #[0.31075111653141246, 0.029414976865931558, -0.000580827233999992, 0.6830127018922193, 0.18301270189221935, 0.6830127018922193, -0.18301270189221935]
    #[0.5, -0.234, 0.206, 0.707, 0.00417, -0.707, 0.00406]
    
def main():
    setup_curobo_logger("warn")
    
    # Predefined cfg file with sample poses, robot cfg, and environment cfg
    cfg_file = load_yaml(join_path(get_world_configs_path(), "sample_curobo_config_shelf_to_sink_UPDATE.yaml"))
    
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
        hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 0, 0, 0, 0])
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
    add_extensions(simulation_app)

    usd_help.load_stage(my_world.stage) # loads Isaac Sim world to Curobo environment
                                        # This effectively creates a shared stage between Isaac Sim and Curobo
    usd_help.add_world_to_stage(world_cfg, base_frame="/World") # Adds collision objects defined in yaml file
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()
    
    my_world.reset()
    robot._articulation_view.initialize()
    idx_list = [robot.get_dof_index(x) for x in j_names]
    robot.set_joint_positions(default_config, idx_list)
    
    # Joint states of the robot in the simulation
    sim_js = robot.get_joints_state()
    sim_js_names = robot.dof_names
    
    
    cu_js = JointState(
        position=tensor_args.to_device(sim_js.positions),
        velocity=tensor_args.to_device(sim_js.velocities),
        acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
        jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
        joint_names=sim_js_names,
    )
    cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

    motion_gen.attach_spheres_to_robot( # +x down, +y right (wrt to ee in default config)
        sphere_tensor=torch.tensor([[0, 0.06, 0.0, 0.06]], dtype=torch.float32),
    )
    sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
    
    spheres = None # spheres for visualizing robot
    while simulation_app.is_running():
        my_world.step(render=True)  # Step the simulation

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
                    
if __name__ == "__main__":
    main()
