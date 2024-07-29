import os
import random
from dot_pattern_generator import create_randomized_grid_dot_pattern

TESTING = False # False if a new SDG is to be generated

# Specify the number of frames to generate
numFrames = 10

# List of texture paths
# Directory containing HDR images
hdr_images_dir = os.path.expanduser('~/isaac_data_generation/assets/hdr_images')

# List all HDR files in the directory
texture_paths = [os.path.join(hdr_images_dir, f) for f in os.listdir(hdr_images_dir) if f.endswith('.hdr')]

if not TESTING:
    # Create randomized dot patterns in png format
    # And save them to /tmp
    # Number of files must be equal to the number of frames
    dotImageSize = (5000, 5000) # Higher resolution for better quality dots
    dotBaseDistance = 40 # The grid distances are randomized within the generator function
    dpi = 300 # Dots per inch for conversion

    for i in range(numFrames):
        dotRadius = random.uniform(2.5, 5) # Randomize the dot radius
        dot_pattern = create_randomized_grid_dot_pattern(dotImageSize, dotBaseDistance, dotRadius, dpi)
        dot_pattern.save(f"/tmp/randomized_grid_dot_pattern_{i}.png")
        print(f"Dot pattern {i} created")

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": True})

from itertools import cycle
import carb.settings
import omni.client
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Gf, PhysxSchema, UsdLux, UsdPhysics, UsdGeom, Sdf
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, DynamicSphere, DynamicCapsule, DynamicCone
from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics

import omni.isaac.core.utils.prims as prims_utils

from omni.isaac.core import PhysicsContext
import synthetic_data_generation.scripts.VisualTactileSensorWriter as VisualTactileSensorWriter
import numpy as np
from scipy.spatial.transform import Rotation as R

class SDGDemo:

    STAGE_SETUP = os.path.expanduser("~/isaac_data_generation/scenes/fv_sdg_setup1.usd")
    FV_PATH = "/World/FV"
    LEFT_FINGER_PATH = "/World/FV/FV_02"
    RIGHT_FINGER_PATH = "/World/FV/FV_03"
    LEFT_CAMERA_PATH = "/World/FV/FV_02/Cube_03/FV_CAM2"
    RIGHT_CAMERA_PATH = "/World/FV/FV_03/Cube_03/FV_CAM3"
    TARGET_PATH = "/World/FV/Target"
    GRIPPER_SKIN_PATH = "/World/Looks/OmniPBR"
    DOME_LIGHT_PATH ="/Environment/DomeLight"
    
    def __init__(self):
        self._timeline = None
        self._stage = None
        self._num_frames = 100
        self._frame_counter = 0
        self._writer = None
        self._render_products = []
        self._timeline_sub = None
        self._stage_event_sub = None
        self._in_running_state = False
        self._output_dir = None
        
        # GetPrimAtPath objects
        self._target_object = None
        self._left_camera = None
        self._right_camera = None
        self._initialLeftCamPos = None
        self._initialRightCamPos = None
        self._FV = None
        self._initialLeftFingerPos = None
        self._initialRightFingerPos = None
        
    def start(
        self,
        num_frames=10,
        out_dir=None,
        env_interval=3,
        use_temp_rp=False,
        seed=None,
    ):
        print("[FingerVisionSDG] Starting")
        if seed is not None:
            random.seed(seed)
        self._num_frames = num_frames
        self._out_dir = out_dir if out_dir is not None else os.path.join(os.getcwd(), "_out_results")
        self._use_temp_rp = use_temp_rp
        self._frame_counter = 0
        self._trigger_distance = 2.0
        self._initFVposition = None
        self._initFVorientation = None
        self._initTargetPos = None

        self._load_env()
        self._setup_sdg()
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline.play()
        
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.CURRENT_TIME_TICKED), self._on_timeline_event
        )
        
        self._stage_event_sub = (
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop_by_type(int(omni.usd.StageEventType.CLOSING), self._on_stage_closing_event)
        )
        
        self._in_running_state = True

    def clear(self):
        self._timeline = None
        self._frame_counter = 0
        self._target_object = None
        if self._stage_event_sub:
            self._stage_event_sub.unsubscribe()
        self._stage_event_sub = None
        if self._timeline_sub:
            self._timeline_sub.unsubscribe()
        self._timeline_sub = None
        self._destroy_render_products()
        self._stage = None
        self._in_running_state = False
        self._left_camera = None
        self._right_camera = None
        self._initialLeftCamPos = None
        self._initialRightCamPos = None
        self._FV = None
        self._initFVposition = None
        self._initFVorientation = None
        # self._initTargetPos = None
        self._initialLeftFingerPos = None
        self._initialRightFingerPos = None
        
    def is_running(self):
        return self._in_running_state

    def _on_stage_closing_event(self, e: carb.events.IEvent):
        self.clear()

    def _load_env(self):
        # Load pre-made stage with FingerVision-like sensor
        omni.usd.get_context().open_stage(self.STAGE_SETUP)
        omni.kit.app.get_app().update() # Ensure stage is loaded
        
        self._stage = omni.usd.get_context().get_stage()
        self._add_physics_scene()

        # Load Background
        self._randomize_dome_lights()

        # Load prims for randomization
        self._target_object = self._stage.GetPrimAtPath(self.TARGET_PATH)
        self._initTargetPos = self._target_object.GetAttribute("xformOp:translate").Get()

        self._FV = self._stage.GetPrimAtPath(self.FV_PATH)    
        
        self._left_camera = self._stage.GetPrimAtPath(self.LEFT_CAMERA_PATH)
        self._right_camera = self._stage.GetPrimAtPath(self.RIGHT_CAMERA_PATH)
        self._initialLeftCamPos = self._left_camera.GetAttribute("xformOp:translate").Get()
        self._initialRightCamPos = self._right_camera.GetAttribute("xformOp:translate").Get()
        
        self._initFVposition = self._FV.GetAttribute("xformOp:translate").Get()
        self._initFVorientation = self._FV.GetAttribute("xformOp:orient").Get()
        self._initialLeftFingerPos = self._stage.GetPrimAtPath(self.LEFT_FINGER_PATH).GetAttribute("xformOp:translate").Get()
        self._initialRightFingerPos = self._stage.GetPrimAtPath(self.RIGHT_FINGER_PATH).GetAttribute("xformOp:translate").Get()

    def _randomize_fv_pose(self):
        # Gripper Parent Prim (FV) randomization
        fv_loc = self._initFVposition + (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(1, 3))
        self._FV.GetAttribute("xformOp:translate").Set(fv_loc)
        omni.kit.app.get_app().update() # Ensure stage is loaded

        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(0, 2*np.pi)
        
        rotation = R.from_rotvec(angle * axis)
        
        fv_rot = rotation.as_quat()
        fv_rot_gf = Gf.Quatd(fv_rot[0], Gf.Vec3d(fv_rot[1], fv_rot[2], fv_rot[3]))

        self._FV.GetAttribute("xformOp:orient").Set(fv_rot_gf)
        
        # Gripper Finger randomization
        leftFinger = self._stage.GetPrimAtPath(self.LEFT_FINGER_PATH)
        rightFinger = self._stage.GetPrimAtPath(self.RIGHT_FINGER_PATH)
        
        leftFingerOffset = (random.uniform(-1, 0), 0, 0)
        rightFingerOffset = (random.uniform(0, 1), 0, 0)
        leftFinger.GetAttribute("xformOp:translate").Set(self._initialLeftFingerPos+ leftFingerOffset)
        rightFinger.GetAttribute("xformOp:translate").Set(self._initialRightFingerPos + rightFingerOffset)
        
    def _randomize_target_pose(self):
        target_offset = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        self._target_object.GetAttribute("xformOp:translate").Set(self._initTargetPos + target_offset)
    
    def _randomize_dome_lights(self):
        domeLight = self._stage.GetPrimAtPath(self.DOME_LIGHT_PATH)
        
        texture_attr = domeLight.GetAttribute("inputs:texture:file")
        if texture_attr:
            texture_attr.Clear() # Remove the current texture
        
        selected_texture = random.choice(texture_paths)
        
        domeLight.GetAttribute("inputs:texture:file").Set(selected_texture)
        
        print(f"Dome light texture path set to: {selected_texture}")
    
    def _randomize_camera_param(self):
        # Randomize camera position for fisheye effect randomization
        leftCamPos = self._initialLeftCamPos + (random.uniform(-0.1, 0), 0, 0)
        self._left_camera.GetAttribute("xformOp:translate").Set(leftCamPos)
        
        rightCamPos = self._initialRightCamPos + (random.uniform(-0.1, 0), 0, 0)
        self._right_camera.GetAttribute("xformOp:translate").Set(rightCamPos)

    def _randomize_dot_patterns(self):
        # Basically loading the saved dot patterns and applying them skin material prim
        mtl_prim = self._stage.GetPrimAtPath(self.GRIPPER_SKIN_PATH)
        
        omni.usd.create_material_input(
            mtl_prim,
            "opacity_texture",
            os.path.expanduser(f"/tmp/randomized_grid_dot_pattern_{self._frame_counter}.png"),
            Sdf.ValueTypeNames.Asset,
        )

    def _add_physics_scene(self):
        # Physics setup specific for the navigation graph
        physics_scene = UsdPhysics.Scene.Define(self._stage, "/physicsScene")
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(self._stage.GetPrimAtPath("/physicsScene"))
        physx_scene.GetEnableCCDAttr().Set(True)
        physx_scene.GetEnableGPUDynamicsAttr().Set(False)
        physx_scene.GetBroadphaseTypeAttr().Set("MBP")
        
        
    def _setup_sdg(self):
        # Disable capture on play and async rendering
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
        carb.settings.get_settings().set("/app/asyncRendering", False)

        self._writer = rep.WriterRegistry.get("VisualTactileSensorWriter")
        self._writer.initialize(output_dir=self._out_dir, 
                                rgb=True, 
                                semantic_segmentation=True, 
                                filter_class=["dot1", "dot2"],
                                object_inview = "inview_obj",
                                )
        # If no temporary render products are requested, create them once here and destroy them only at the end
        if not self._use_temp_rp:
            self._setup_render_products()

    def _setup_render_products(self):
        print(f"[FingerVisionSDG] Creating render products")
        
        rp_left = rep.create.render_product(
            self.LEFT_CAMERA_PATH,
            (1920, 1080),
            name="left_finger",
        )
        
        rp_right = rep.create.render_product(
            self.RIGHT_CAMERA_PATH,
            (1920, 1080),
            name="right_finger",
        )
        self._render_products = [rp_left, rp_right]
        self._writer.attach(self._render_products)
        rep.orchestrator.preview()

    def _destroy_render_products(self):
        print(f"[FingerVisionSDG] Destroying render products")
        if self._writer:
            self._writer.detach()
        for rp in self._render_products:
            rp.destroy()
        self._render_products.clear()

    def _run_sdg(self):
        if self._use_temp_rp:
            self._setup_render_products()
        rep.orchestrator.step(rt_subframes=16, pause_timeline=False)
        rep.orchestrator.wait_until_complete()
        if self._use_temp_rp:
            self._destroy_render_products()

    def _on_sdg_done(self, task):
        self._setup_next_frame()
    
    def _delete_spawn_target(self):
        if self._stage.GetPrimAtPath(self.TARGET_PATH):
            omni.kit.commands.execute("DeletePrimsCommand", paths=[self.TARGET_PATH])
        
        object_types = [DynamicCuboid, DynamicCapsule, DynamicCone, DynamicSphere, DynamicCylinder]
        selected_object = random.choice(object_types)
        
        selected_object(prim_path=self.TARGET_PATH,
            position=self._initTargetPos,
            scale = np.array([1, 1, 1]),
            color = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]),
            )
        
        self._target_object = self._stage.GetPrimAtPath(self.TARGET_PATH)
        
        add_update_semantics(self._target_object, semantic_label="inview_obj", type_label="class")

    def _setup_next_frame(self):
        self._frame_counter += 1
        if self._frame_counter >= self._num_frames:
            print("SDG Finished")
            self.clear()
            return
        
        # Spawn new target object shape
        self._delete_spawn_target()
        # Randomize dot patterns, gripper pose, target pose, camera parameters, and background
        self._randomize_dot_patterns()
        self._randomize_fv_pose()
        self._randomize_target_pose()
        self._randomize_camera_param()
        self._randomize_dome_lights()

        self._timeline.play()
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.CURRENT_TIME_TICKED), self._on_timeline_event
        )

    def _on_timeline_event(self, e: carb.events.IEvent):
        print("Timeline event")
                
        if True: #CONDITION TO CAPTURE FRAME
            print(f"[FingerVisionSDGDemo] Capturing frame no. {self._frame_counter}")
            self._timeline.pause()
            self._timeline_sub.unsubscribe()

            self._run_sdg()
            self._setup_next_frame()

# parser = argparse.ArgumentParser()
# parser.add_argument("--use_temp_rp", action="store_true", help="Create and destroy render products for each SDG frame")
# parser.add_argument("--num_frames", type=int, default=9, help="The number of frames to capture")
# parser.add_argument("--env_interval", type=int, default=3, help="Interval at which to change the environments")
# args, unknown = parser.parse_known_args()

out_dir = os.path.join(os.getcwd(), os.path.expanduser("~/isaac_data_generation/images/synthetic_data"), "")
nav_demo = fvSDGDemo()
nav_demo.start(
    num_frames=numFrames,
    out_dir=out_dir,
    env_interval=10,
    use_temp_rp=False,
    seed=124,
)

while simulation_app.is_running() and nav_demo.is_running():
    simulation_app.update()

simulation_app.close()
