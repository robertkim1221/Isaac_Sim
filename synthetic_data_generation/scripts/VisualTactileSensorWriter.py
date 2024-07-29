from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch
import omni.replicator.core.functional as F
import omni.replicator.core as rep
from omni.replicator.core.bindings._omni_replicator_core import Schema_omni_replicator_extinfo_1_0
from omni.isaac.core.utils import prims
import numpy as np

__version__ = "0.0.1"


class VisualTactileSensorWriter(Writer):
    def __init__(
        self,
        output_dir,
        semantic_segmentation=False,
        image_output_format="jpeg",
        frame_padding=4,
        colorize_semantic_segmentation=False,
        rgb=True,
        filter_class=None,
        object_inview="",
        occlusion=True,
    ):
        self._output_dir = output_dir
        self._backend = BackendDispatch(output_dir=output_dir)
        self.backend = self._backend
        self._frame_id = 0
        self._sequence_id = 0
        self._image_output_format = image_output_format
        self._frame_padding = frame_padding
        self.annotators = []
        self.colorize_semantic_segmentation = colorize_semantic_segmentation
        self.render_product_name = []
        self.filter_class = filter_class if filter_class else []
        self.target_instance_id = None
        self.version = __version__
        self._telemetry = Schema_omni_replicator_extinfo_1_0()
        self.object_inview = object_inview
        
        if semantic_segmentation:
            self.annotators.append(AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": colorize_semantic_segmentation}))
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        if occlusion:
            self.annotators.append(AnnotatorRegistry.get_annotator("occlusion"))

    def write(self, data: dict):
        sequence_id = ""
        for trigger_name, call_count in data["trigger_outputs"].items():
            if "on_time" in trigger_name:
                sequence_id = f"{call_count}_{sequence_id}"
        if sequence_id != self._sequence_id:
            self._frame_id = 0
            self._sequence_id = sequence_id

        for annotator in data.keys():
            annotator_split = annotator.split("-")
            render_product_path = ""
            multi_render_prod = 0
            if len(annotator_split) > 1:
                multi_render_prod = 1
                render_product_name = annotator_split[-1]
                render_product_path = f"{render_product_name}/"
                if render_product_name not in self.render_product_name:
                    self.render_product_name.append(render_product_name)

                try:
                    index = self.render_product_name.index(render_product_name)
                    filter_class = self.filter_class[index]
                except ValueError:
                    filter_class = None
                    print(f"Render product name: {render_product_name} not found in list")
                    continue

            if annotator.startswith("semantic_segmentation"):
                if multi_render_prod:
                    render_product_path += "semantic_segmentation/"
                    object_path = render_product_path + "inview_obj/"
                self._write_semantic_segmentation(data, render_product_path, annotator, filter_class)

                self._write_inview_segmentation(data, object_path, annotator, self.object_inview)

            if annotator.startswith("rgb"):
                if multi_render_prod:
                    render_product_path += "rgb/"
                self._write_rgb_segmentation(data, render_product_path, annotator)

        self._frame_id += 1

    def get_instance_id_for_class(self, id_to_labels, target_class):
        for instance_id, label_info in id_to_labels.items():
            classes = label_info['class'].split(',')
            if target_class in classes:
                return int(instance_id)
        return None

    def _write_semantic_segmentation(self, data: dict, render_product_path: str, annotator: str, filter_class: str):
        semantic_seg_data = data[annotator]["data"]
        height, width = semantic_seg_data.shape[:2]

        id_to_labels = data[annotator]["info"]["idToLabels"]

        self.target_instance_id = self.get_instance_id_for_class(id_to_labels, filter_class)
        if self.target_instance_id is None:
            print(f"No instance ID found for class {filter_class}")
            return

        binary_mask = np.zeros((height, width), dtype=np.uint8)
        self.target_instance_id = np.uint32(self.target_instance_id)
        binary_mask = np.where(semantic_seg_data == self.target_instance_id, 255, 0).astype(np.uint8)
        found = np.sum(binary_mask == 255)
        print(f"Found {found} pixels for instance ID {self.target_instance_id}")

        file_path = (
            f"{render_product_path}semantic_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.{self._image_output_format}"
        )
        if self.colorize_semantic_segmentation:
            self._backend.schedule(F.write_image, data=binary_mask, path=file_path)
        else:
            self._backend.schedule(F.write_image, data=binary_mask, path=file_path)

    def _write_inview_segmentation(self, data: dict, render_product_path: str, annotator: str, filter_class: str):
        semantic_seg_data = data[annotator]["data"]

        height, width = semantic_seg_data.shape[:2]

        id_to_labels = data[annotator]["info"]["idToLabels"]

        self.target_instance_id = self.get_instance_id_for_class(id_to_labels, filter_class)
        if self.target_instance_id is None:
            print(f"No instance ID found for class {filter_class}")
            return

        binary_mask = np.zeros((height, width), dtype=np.uint8)
        self.target_instance_id = np.uint32(self.target_instance_id)
        binary_mask = np.where(semantic_seg_data == self.target_instance_id, 255, 0).astype(np.uint8)
        found = np.sum(binary_mask == 255)
        print(f"Found {found} pixels for instance ID {self.target_instance_id}")

        file_path = (
            f"{render_product_path}inview_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.{self._image_output_format}"
        )
        if self.colorize_semantic_segmentation:
            self._backend.schedule(F.write_image, data=binary_mask, path=file_path)
        else:
            self._backend.schedule(F.write_image, data=binary_mask, path=file_path)

    def _write_rgb_segmentation(self, data: dict, render_product_path: str, annotator: str):
        file_path = f"{render_product_path}rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.{self._image_output_format}"
        self._backend.schedule(F.write_image, data=data[annotator], path=file_path)

    def on_final_frame(self):
        print(f"Final frame {self._frame_id} reached")

rep.WriterRegistry.register(VisualTactileSensorWriter)
