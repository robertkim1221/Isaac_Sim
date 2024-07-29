import numpy as np
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.prims import RigidPrim

########## Curobo Initialization ###########
from CuroboMotionPlanner import CuroboMotionPlanner
from helper import add_extensions, add_robot_to_scene
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_world_configs_path

config_file = "curobo_config_20240729_fixed.yaml"

dish_file = "dish_yml"
curoboMotion = CuroboMotionPlanner(config_file)

world_cfg, robot_cfg = curoboMotion.load_stage()
curoboMotion.setup_motion_planner()

dish_cfg_path = get_world_configs_path() + "/" + "dish.yml"

########## ISAAC SIM INITIALIZATION ###########
sim_world = World(stage_units_in_meters=1.0)
stage = sim_world.stage
xform = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(xform) 
stage.DefinePrim("/curobo", "Xform")
# Interfacing Isaac Sim with CuRobo
add_extensions(simulation_app)

usd_help = UsdHelper()

usd_help.load_stage(sim_world.stage) # loads Isaac Sim world to Curobo environment
                                    # This effectively creates a shared stage between Isaac Sim and Curobo
usd_help.add_world_to_stage(world_cfg, base_frame="/World") # Adds collision objects defined in yaml file
robot, robot_prim_path = add_robot_to_scene(robot_cfg, sim_world)
articulation_controller = robot.get_articulation_controller()

i = 0 # Counter for waiting for user to click play

default_pose = curoboMotion.default_config
# target_pose1 = [0.5031068136513849, -0.1624755148641493, 0.1, -0.18301270189221927, 0.6830127018922193,-0.18301270189221927, -0.6830127018922193]
# target_poses = [target_pose1, default_pose]

target_list = curoboMotion.pose_list
pose_idx = 0

ee_prim_path = "/World/ur5e_robot/ee"
ee_prim = RigidPrim(ee_prim_path)
default_js = curoboMotion.default_config
target_reached = False
spheres = None

action_queue = []

target_pose = curoboMotion.remap_pose(target_list[pose_idx])
reset = False
visualize_isaac_sim = True
obj_attach = True
ignore_prim_paths = [robot_prim_path, "/curobo"]

while simulation_app.is_running():
    # print(f"Target pose idx {pose_idx}")
    # print(f"Total number of target poses {len(target_list)}")
    if pose_idx >= len(target_list):
        print("All target poses reached")
        break
    
    sim_world.step(render=True)
    
    if not sim_world.is_playing(): # Wait for user to click play
        if i % 100 == 0:
            print("**** Click Play to start simulation *****")
        i += 1
        continue
    
    step_index = sim_world.current_time_step_index   # Get current time step index

    if step_index < 2: # Reset the simulation and robot to initial state
        sim_world.reset()
        robot._articulation_view.initialize()
        idx_list = [robot.get_dof_index(x) for x in curoboMotion.j_names]
        robot.set_joint_positions(curoboMotion.default_config, idx_list)

        robot._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
        )
        
    if step_index < 20: # Allow the robot/sim to settle
        continue
    
    if step_index % 500 == 0:
        obstacles = usd_help.get_obstacles_from_stage(
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore_prim_paths,
        ).get_collision_check_world()
        curoboMotion.motion_gen.update_world(obstacles)
    

    curoboMotion.attach_obj(dish_cfg_path)
            # print("detaching object")
    # # Check whether goal has been reached
    # ee_position, ee_orientation = ee_prim.get_world_pose()
    # ee_orientation = np.array([ee_orientation[3], ee_orientation[0], ee_orientation[1], ee_orientation[2]])
    # norm_pos = np.linalg.norm(np.array(ee_position) - np.array(target_pose[:3]))
    # norm_ori = np.linalg.norm(np.array(ee_orientation) - np.array(target_pose[3:]))
    
    # if (norm_pos < 0.02) and (norm_ori < 0.02):
    #     target_reached = True
    # else:
    #     target_reached = False    
    # Joint states of the robot in the simulation
    

    sim_js = robot.get_joints_state()
    sim_js_names = robot.dof_names
    cu_js = curoboMotion.get_js_isaac_sim(robot_prim=robot)
    robot_static = np.max(np.abs(sim_js.velocities)) < 0.2
    
    assert not np.any(np.isnan(sim_js.positions)), "isaac sim has returned NAN joint position values."
    
    if robot_static:
        cu_js = curoboMotion.get_js_isaac_sim(robot_prim=robot)
        
        solution = curoboMotion.motion_planner(
            initial_js=cu_js.position.cpu().numpy(),
            goal_ee_pose=target_pose,
        )
        # if target_pose != default_pose:
        #     solution = curoboMotion.motion_planner(
        #         initial_js=cu_js.position.cpu().numpy(),
        #         goal_ee_pose=target_pose,
        #     )
        # else:
        #     solution = curoboMotion.motion_planner(
        #         initial_js=cu_js.position.cpu().numpy(),
        #         goal_js=default_pose,
        #     )
            
        if solution is None:
            print("Failed to find a solution")
            pose_idx += 1

            target_pose = curoboMotion.remap_pose(target_list[pose_idx])

            continue
        else:
            print(solution["collision_distances"])

        
        # if not action_queue:
        #     actions = curoboMotion.get_isaac_sim_action(solution, robot_prim=robot)

        #     action_queue.extend(actions)
        # else:
        #     next_action = action_queue.pop(0)
        #     articulation_controller.apply_action(next_action)
        #     sim_world.step(render=False)
       
        # if action_queue:
        #     next_action = action_queue.pop(0)
        #     articulation_controller.apply_action(next_action)

        if visualize_isaac_sim: 
            actions = curoboMotion.get_isaac_sim_action(solution, robot_prim=robot)

            for action_idx, action in enumerate(actions):
                articulation_controller.apply_action(action)
                cu_js = curoboMotion.get_js_isaac_sim(robot_prim=robot)

                curoboMotion.visualize_spheres_and_collision_vector(cu_js.position.cpu().numpy().tolist())
    
                sim_world.step(render=True)
    
        robot.set_joint_positions(curoboMotion.default_config, idx_list)

    pose_idx += 1
    target_pose = curoboMotion.remap_pose(target_list[pose_idx])
    print("next target pose")
    reset = False
# target_pose = target_poses[current_target_index]
        
simulation_app.close()