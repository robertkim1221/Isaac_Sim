import torch
# CuRobo
# Standard Library
from typing import Dict, List, Optional

# Third Party
import carb
import json
import numpy as np
from curobo.geom.sdf.world import CollisionCheckerType

from curobo.geom.types import WorldConfig, Cuboid, Cylinder
# from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util_file import (
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

import yaml

class CuroboMotionPlanner:
    def __init__(self, config_file: str):
        
        self.cfg_file = load_yaml(join_path(get_world_configs_path(), config_file))
        self.robot_file = load_yaml(join_path(get_robot_configs_path(), self.cfg_file["config_file"]))

        self.robot_cfg = self.robot_file["robot_cfg"]
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"] 
        self.q_start = self.cfg_file["q_start"]
               
        self.pose_list = self.cfg_file["x_desired"]
        self.pose_descriptors = self.cfg_file["pose_descriptors"]
        self.cubes = []
        self.world_cfg = None

        self.detach = None

        self.motion_gen = None
        self.plan_config = None
        # Warmup CuRobo instance
        self.tensor_args = TensorDeviceType()
        
        # Iteration Variables
        self.cmd_plan = None
        self.cmd_state = None
        self.cmd_idx = 0
        
        self.spheres = None
    
    def load_stage(self):
        # Load Environment Configuration from yaml
        env = self.cfg_file['env']
        for obj in env:
            self.cubes.append(Cuboid(name=obj['name'], pose=self.remap_pose(obj['pose']), dims=obj['dim']))
        self.world_cfg = WorldConfig(cuboid=self.cubes)
        
        return self.world_cfg, self.robot_cfg     
        
    def setup_motion_planner(self):
        # Configure MotionGen
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            # collision_checker_type=CollisionCheckerType.MESH,
            # num_trajopt_seeds=24,
            # num_graph_seeds=12,
            # interpolation_dt=0.05,
            # collision_cache={"obb": 50, "mesh": 100},
            # optimize_dt=True,
            # trajopt_dt=None,
            # trajopt_tsteps=32,
            # trim_steps=None,
            # num_ik_seeds=32,
            # js_trajopt_dt=0.15,
            # project_pose_to_goal_frame=False
            interpolation_dt=0.01,
            trajopt_tsteps=24,
            use_cuda_graph=False,
            project_pose_to_goal_frame=False,
            # collision_activation_distance=0.005,
            
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("Curobo warming up...")
        self.motion_gen.warmup(enable_graph=False, warmup_js_trajopt=True, parallel_finetune=True)
        print("CuRobo is Ready")
        
        pose_cost_metric = PoseCostMetric(
            hold_partial_pose = True,
            hold_vec_weight=self.motion_gen.tensor_args.to_device([1, 1, 0, 0, 0, 0]),
            reach_partial_pose=True,
            reach_vec_weight=self.motion_gen.tensor_args.to_device([1, 1, 1, 1, 1, 1]),
        )

        
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=None,
            max_attempts=100,
            enable_finetune_trajopt=True,
            partial_ik_opt=True,
            parallel_finetune=True,
            pose_cost_metric=pose_cost_metric,
        )
        
    @staticmethod
    def convert_joint_state_to_goal(pos, vel=None, accel=None, n=1):
        if not vel:
            vel = [0.0] * len(pos)

        if not accel:
            accel = [0.0] * len(pos)

        state = JointState(
            torch.Tensor([pos] * n).float().cuda(),
            torch.Tensor([vel] * n).float().cuda(),
            torch.Tensor([accel] * n).float().cuda(),
            tensor_args=TensorDeviceType()
        )

        return state
    
    def setup_collision_checker(self, cu_js_position: List[float]):
        act_distance = 0.05
        config = RobotWorldConfig.load_from_config(
            self.robot_file,
            self.world_cfg,
            collision_activation_distance=act_distance,
            collision_checker_type=CollisionCheckerType.MESH,
        )
        
        model = RobotWorld(config)
        sph_list = self.motion_gen.kinematics.get_robot_as_spheres(self.tensor_args.to_device(cu_js_position))
        # Find the spheres associated with the specified link
        last_sphere = sph_list[0][-1]

        x_sph = torch.zeros((1, 1, 1, 4), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        x_sph[..., :3] = self.tensor_args.to_device(last_sphere.position).view(1, 1, 1, 3)
        x_sph[..., 3] = torch.tensor(last_sphere.radius, device=self.tensor_args.device, dtype=self.tensor_args.dtype)

        d, d_vec = model.get_collision_vector(x_sph)

        # Print the collision vector and distance
        print(f"Collision Distance for the last sphere at {last_sphere.position}: {d.item()}")
        print(f"Collision Vector for the last sphere at {last_sphere.position}: {d_vec.cpu().numpy()}")

        return last_sphere.position, d, d_vec
        
        
    @staticmethod
    def remap_pose(xyzw_pose: List[float]) -> List[float]:
        return [xyzw_pose[i] for i in [0, 1, 2, 6, 3, 4, 5]] # for xyzw to wxyz
    
    @staticmethod
    def load_dish_tensors(file_path):
        with open(file_path, "r") as file:
            dish_cfg = yaml.safe_load(file)
        
        dish_spheres = []
        for dish in dish_cfg["dish"]:
            position = dish["position"]
            radius = dish["radius"]
            sphere_tensor = [position[0], position[1], position[2], radius]
            dish_spheres.append(sphere_tensor)
        return dish_spheres

    @staticmethod
    def convert_xyzw_poses_to_curobo(poses):
        poses = np.array(poses)
        pos = torch.from_numpy(poses[:,:3].copy())
        quat = torch.from_numpy(poses[:,[6,3,4,5]].copy())

        pose = Pose(
            position=pos.float().cuda(),
            quaternion=quat.float().cuda(),
        )

        return pose

# TODO: FIGURE OUT A WAY TO MAKE BATCH SIZE BASED ON POSE_DESCRIPTORS ENDING IN THE SAME NUMBER AND PLAN ACCORDINGLY
# TODO: FOR 10CM BOWLS, MAYBE A SIMPLE SPHERE IS NOT THE WAY TO GO; MAYBE A CYLINDER? OR HAVE MULTIPLE SMALL SPHERES INSTEAD OF ONE BIG SPHERE
# TODO: FIND OUT IF I CAN IMPORT MESHES INSTEAD OF SPHERES FOR COLLISION CHECKING. IF NOT CYLINDER IS BEST BET

    def batch_motion_planner(self,
                             goal_ee_poses: Optional[List[List[float]]] = None,
                             batch_size: int = 16,
                             ):
        start_state = self.convert_joint_state_to_goal(self.q_start, n=batch_size)
        pose_descriptors = self.pose_descriptors
        
        dish_radii = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        solutions = []

        for radius in dish_radii:
            print("Dish Radius: ", radius)
            start_i = 0
            solutions = []
            self.motion_gen.detach_spheres_from_robot()
            dish_sphere_tol = 0.01
            dish_tensor = torch.tensor([[0, radius-dish_sphere_tol, 0.0, radius-dish_sphere_tol]], dtype=torch.float32)
            self.motion_gen.attach_spheres_to_robot(sphere_tensor=dish_tensor)
            
            while start_i < len(goal_ee_poses):
                poses = goal_ee_poses[start_i : start_i + batch_size]
                if len(poses) < batch_size:
                    poses.extend([poses[-1]] * (batch_size - len(poses)))

                end_state = self.convert_xyzw_poses_to_curobo(poses)
                res = self.motion_gen.plan_batch(start_state, end_state, plan_config=self.plan_config)
                
                success = res.success.cpu().numpy()
                print("Successes: {} out of {}".format(np.sum(success), len(success)))
                if np.any(success):
                    dts = res.optimized_dt.cpu().numpy().tolist()
                    trajs = res.optimized_plan.position.cpu().numpy()
                    
                for rel_i in range(batch_size):
                    abs_i = start_i + rel_i
                    if abs_i >= len(goal_ee_poses):
                        break
                    
                    pose = np.array(poses[rel_i]).tolist()
                    if not success[rel_i]:
                        traj = []
                        dt = 0.0
                    else:
                        traj = trajs[rel_i].tolist()
                        dt = dts[rel_i]
                    
                    descriptor = pose_descriptors[abs_i]
                    
                    solutions.append({
                        "pose": pose,
                        "traj": traj,
                        "dt": dt,
                        "descriptor": descriptor,
                    })

                start_i += batch_size
                print ("{} out of {}".format(start_i, len(goal_ee_poses)))
                            
        with open("interpolated_plans.json", "w") as file:
                json.dump(solutions, file, indent=4)
                    
                           
    def motion_planner(self,
                       initial_js: List[float],
                       goal_ee_pose: Optional[List[float]] = None,
                       goal_js: Optional[List[float]] = None,
    ):
        if goal_ee_pose is not None and goal_js is None:
            initial_js = JointState.from_position(
                position=self.tensor_args.to_device([initial_js]),
                joint_names=self.j_names[0 : len(initial_js)],
            )
            
            goal_pose = Pose(
                position=self.tensor_args.to_device(goal_ee_pose[0:3]),
                quaternion=self.tensor_args.to_device(goal_ee_pose[3:]),
            )
            
            try:
                motion_gen_result = self.motion_gen.plan_single(
                    initial_js, goal_pose, self.plan_config
                )
                reach_succ = motion_gen_result.success.item()

            except Exception as e:
                return None
            
        elif goal_js is not None and goal_ee_pose is None:
            initial_js = JointState.from_position(
                position=self.tensor_args.to_device([initial_js]),
                joint_names=self.j_names[0 : len(initial_js)],
            )        
            
            goal_js = JointState.from_position(
                position=self.tensor_args.to_device([goal_js]),
                joint_names=self.j_names[0 : len(goal_js)],
            )
            
            try:
                motion_gen_result = self.motion_gen.plan_single_js(
                    initial_js, goal_js, self.plan_config
                )
                reach_succ = motion_gen_result.success.item()

            except:
                return None
        else:
            raise ValueError("Either goal_js or goal_ee_pose must be provided.")
        
        if reach_succ:
            interpolated_solution = motion_gen_result.get_interpolated_plan()
            
            collision_distances = []
            for joint_position in interpolated_solution.position.cpu().numpy():
                _, d, _ = self.setup_collision_checker(joint_position.tolist())
                collision_distances.append(d.item())    
        
            solution_dict = {
                "success": motion_gen_result.success.item(),
                "joint_names": interpolated_solution.joint_names,
                "positions": interpolated_solution.position.cpu().squeeze().numpy().tolist(),
                "velocities": interpolated_solution.velocity.cpu().squeeze().numpy().tolist(),
                "accelerations": interpolated_solution.acceleration.cpu().squeeze().numpy().tolist(),
                "jerks": interpolated_solution.jerk.cpu().squeeze().numpy().tolist(),
                "interpolation_dt": motion_gen_result.interpolation_dt,
                "collision_distances": collision_distances,
                "raw_data": interpolated_solution,
            }
            
            return solution_dict
        else:
            return None
       
    def attach_obj(self, dish_file_path: str) -> None:
        self.detach = False
        collision_spheres = self.load_dish_tensors(dish_file_path)
        sphere_tensor = collision_spheres[0]
        self.motion_gen.attach_spheres_to_robot(sphere_tensor=torch.tensor([sphere_tensor], dtype=torch.float32))
        
    def detach_obj(self) -> None:
        self.detach = True
        self.motion_gen.detach_spheres_from_robot()
    
    def visualize_spheres_and_collision_vector(self, cu_js_position: List[float]):
        from omni.isaac.core.objects import sphere

        sph_list = self.motion_gen.kinematics.get_robot_as_spheres(self.tensor_args.to_device(cu_js_position))
        num_spheres = len(sph_list[1])
        
        # Clear spheres if detach is True
        if self.detach:
            self.spheres = None
        if self.spheres is None:
            self.spheres = []
            # create spheres:
            for si, s in enumerate(sph_list[0]):
                sp = sphere.VisualSphere(
                    prim_path="/curobo/robot_sphere_" + str(si),
                    position=np.ravel(s.position),
                    radius=float(s.radius),
                    color=np.array([0, 0.8, 0.2]),
                )
                self.spheres.append(sp)
        else:
            for si in range(len(self.spheres), num_spheres):
                sp = sphere.VisualSphere(
                    prim_path="/curobo/robot_sphere_" + str(si),
                    position=np.ravel(sph_list[0][si].position),
                    radius=float(sph_list[0][si].radius),
                    color=np.array([0, 0.8, 0.2]),
                )
                self.spheres.append(sp)
                
            for si, s in enumerate(sph_list[0]):
                if not np.isnan(s.position[0]):
                    self.spheres[si].set_world_pose(position=np.ravel(s.position))
                    self.spheres[si].set_radius(float(s.radius))
                    
        # Draw collision vector
        sph_position, d, d_vec = self.setup_collision_checker(cu_js_position)
        d = d.view(-1).cpu()
        p = d.item()
        p = max(1, p * 5)
        
        try:        
            from omni.isaac.debug_draw import _debug_draw
        except ImportError:
            raise ImportError("Failed to import _debug_draw. Double check Isaac Sim environment and configuration.")
        
        draw = _debug_draw.acquire_debug_draw_interface()

        if d.item() != 0.0:
            start = sph_position
            gradient = d_vec[..., :3].view(3).cpu().numpy()
            
            draw.clear_lines()
            start_list = [start]
            end_list = [start + gradient]

            colors = [(1, 0, 0, 0.8)]

            sizes = [10.0]
            draw.draw_lines(start_list, end_list, colors, sizes)
        else:
            draw.clear_lines()
        
    def get_js_isaac_sim(self, robot_prim):
        sim_js = robot_prim.get_joints_state()
        sim_js_names = robot_prim.dof_names
        
        if np.any(np.isnan(sim_js.positions)):
            carb.log_warn("Isaac Sim has returned NAN joint position values.")
            
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities),
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        return cu_js
    
    def get_isaac_sim_action(self, solution: dict, robot_prim):
        try:
            from omni.isaac.core.utils.types import ArticulationAction
        except ImportError:
            raise ImportError(
                "Failed to import ArticulationAction. Double check Isaac Sim environment and configuration."
            )
    
        self.cmd_plan = self.motion_gen.get_full_js(
            solution["raw_data"]
        )
        isaac_joint_names = robot_prim.dof_names
              
        isaac_joint_idx = []
        common_js_names = []
        
        for name in isaac_joint_names:
            if name in isaac_joint_names:
                isaac_joint_idx.append(robot_prim.get_dof_index(name))
                common_js_names.append(name)
        
        ordered_js_solution = self.cmd_plan.get_ordered_joint_state(common_js_names) 
        
        positions = ordered_js_solution.position.cpu().numpy()
        velocities = ordered_js_solution.velocity.cpu().numpy()
        actions = []
        
        for i in range(len(positions)):
            actions.append(
                ArticulationAction(
                    joint_positions=positions[i],
                    joint_velocities=velocities[i],
                    joint_indices=isaac_joint_idx,
                )
            )
        return actions
        
