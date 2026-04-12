# Acknowledgement:
#   1. https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/waymo_converter.py
#   2. https://github.com/leolyj/DCA-SRSFE/blob/main/data_preprocess/Waymo/generate_flow.py
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0" '
        ">1.4.5 to install the official devkit first."
    )

import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection

from datasets.tools.multiprocess_utils import track_parallel_progress
from .waymo_utils import (
    parse_range_image_flow_and_camera_projection,
    convert_range_image_to_point_cloud_flow,
    project_vehicle_to_image,
    get_ground_np
)

MOVEABLE_OBJECTS_IDS = [
    cs_pb2.CameraSegmentation.TYPE_CAR,
    cs_pb2.CameraSegmentation.TYPE_TRUCK,
    cs_pb2.CameraSegmentation.TYPE_BUS,
    cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE,
    cs_pb2.CameraSegmentation.TYPE_BICYCLE,
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE,
    cs_pb2.CameraSegmentation.TYPE_TRAILER,
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN,
    cs_pb2.CameraSegmentation.TYPE_CYCLIST,
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST,
    cs_pb2.CameraSegmentation.TYPE_BIRD,
    cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL,
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT,
]

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
# TODO(ziyu): consider all dynamic classes
WAYMO_DYNAMIC_CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']
WAYMO_HUMAN_CLASSES = ['Pedestrian', 'Cyclist']
WAYMO_VEHICLE_CLASSES = ['Vehicle']

# OpenCV to Waymo dataset coordinate transformation.
# opencv coordinate system: x right, y down, z front
# waymo coordinate system: x front, y left, z up
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)

class WaymoProcessor(object):
    """Process Waymo dataset.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        prefix,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects"
        ],
        process_id_list=None,
        workers=64,
    ):
        self.filter_no_label_zone_points = True

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split(".")[0]) < 2:
            tf.enable_eager_execution()

        # keep the order defined by the official protocol
        self.cam_list = [
            "_FRONT",
            "_FRONT_LEFT",
            "_FRONT_RIGHT",
            "_SIDE_LEFT",
            "_SIDE_RIGHT",
        ]
        self.lidar_list = ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]

        self.load_dir = load_dir
        self.save_dir = f"{save_dir}/{prefix}"
        self.workers = int(workers)
        # a list of tfrecord pathnames
        training_files = open("data/waymo_train_list.txt").read().splitlines()
        self.tfrecord_pathnames = [
            f"{self.load_dir}/{f}.tfrecord" for f in training_files
        ]
        # self.tfrecord_pathnames = sorted(glob(join(self.load_dir, "*.tfrecord")))
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type="")
        num_frames = sum(1 for _ in dataset)
        for frame_idx, data in enumerate(
            tqdm(dataset, desc=f"File {file_idx}", total=num_frames, dynamic_ncols=True)
        ):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (
                self.selected_waymo_locations is not None
                and frame.context.stats.location not in self.selected_waymo_locations
            ):
                continue
            if "images" in self.process_keys:
                self.save_image(frame, file_idx, frame_idx)
            if "calib" in self.process_keys:
                self.save_calib(frame, file_idx, frame_idx)
            if "lidar" in self.process_keys:
                self.save_lidar(frame, file_idx, frame_idx)
            if "pose" in self.process_keys:
                self.save_pose(frame, file_idx, frame_idx)
            if "depth_map" in self.process_keys:
                self.save_depth_map(frame, file_idx, frame_idx)
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(frame, file_idx, frame_idx, class_valid='all')
                self.save_dynamic_mask(frame, file_idx, frame_idx, class_valid='human')
                self.save_dynamic_mask(frame, file_idx, frame_idx, class_valid='vehicle')                
            if frame_idx == 0:
                self.save_interested_labels(frame, file_idx)
        if "objects" in self.process_keys:
            instances_info, frame_instances = self.save_objects(dataset)
            
            # Save instances info and frame instances
            object_info_dir = f"{self.save_dir}/{str(file_idx).zfill(3)}/instances"
            with open(f"{object_info_dir}/instances_info.json", "w") as fp:
                json.dump(instances_info, fp, indent=4)
            with open(f"{object_info_dir}/frame_instances.json", "w") as fp:
                json.dump(frame_instances, fp, indent=4)

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_interested_labels(self, frame, file_idx):
        """
        Saves the interested labels of a given frame to a JSON file.

        Args:
            frame: A `Frame` object containing the labels to be saved.
            file_idx: An integer representing the index of the file to be saved.

        Returns:
            None
        """
        frame_data = {
            "time_of_day": frame.context.stats.time_of_day,
            "location": frame.context.stats.location,
            "weather": frame.context.stats.weather,
        }
        object_type_name = lambda x: label_pb2.Label.Type.Name(x)
        object_counts = {
            object_type_name(x.type): x.count
            for x in frame.context.stats.camera_object_counts
        }
        frame_data.update(object_counts)
        # write as json
        with open(
            f"{self.save_dir}/{str(file_idx).zfill(3)}/frame_info.json",
            "w",
        ) as fp:
            json.dump(frame_data, fp)

    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = (
                f"{self.save_dir}/{str(file_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.jpg"
            )
            with open(img_path, "wb") as fp:
                fp.write(img.image)

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        extrinsics = []
        intrinsics = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            intrinsic = list(camera.intrinsic)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            np.savetxt(
                f"{self.save_dir}/{str(file_idx).zfill(3)}/extrinsics/"
                + f"{str(i)}.txt",
                extrinsics[i],
            )
            np.savetxt(
                f"{self.save_dir}/{str(file_idx).zfill(3)}/intrinsics/"
                + f"{str(i)}.txt",
                intrinsics[i],
            )

    def save_lidar(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        (
            range_images,
            camera_projections,
            seg_labels,
            range_image_top_pose,
        ) = parse_range_image_and_camera_projection(frame)

        # https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/segmentation.proto
        if range_image_top_pose is None:
            # the camera only split doesn't contain lidar points.
            return

        # collect first return only
        range_images_flow, _, _ = parse_range_image_flow_and_camera_projection(frame)
        (
            origins,
            points,
            flows,
            cp_points,
            intensity,
            elongation,
            laser_ids,
        ) = convert_range_image_to_point_cloud_flow(
            frame,
            range_images,
            range_images_flow,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
        )
        origins = np.concatenate(origins, axis=0)
        points = np.concatenate(points, axis=0)
        ground_label = get_ground_np(points)
        intensity = np.concatenate(intensity, axis=0)
        elongation = np.concatenate(elongation, axis=0)
        laser_ids = np.concatenate(laser_ids, axis=0)

        #  -1: no-flow-label, the point has no flow information.
        #   0:  unlabeled or "background,", i.e., the point is not contained in a
        #       bounding box.
        #   1: vehicle, i.e., the point corresponds to a vehicle label box.
        #   2: pedestrian, i.e., the point corresponds to a pedestrian label box.
        #   3: sign, i.e., the point corresponds to a sign label box.
        #   4: cyclist, i.e., the point corresponds to a cyclist label box.
        flows = np.concatenate(flows, axis=0)

        point_cloud = np.column_stack(
            (
                origins,
                points,
                flows,
                ground_label,
                intensity,
                elongation,
                laser_ids,
            )
        )
        pc_path = (
            f"{self.save_dir}/"
            + f"{str(file_idx).zfill(3)}/lidar/{str(frame_idx).zfill(3)}.bin"
        )
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            f"{self.save_dir}/{str(file_idx).zfill(3)}/ego_pose/"
            + f"{str(frame_idx).zfill(3)}.txt",
            pose,
        )

    def save_depth_map(self, frame, file_idx, frame_idx):
        """Project Waymo LiDAR points to each camera and save sparse depth maps."""
        (
            range_images,
            camera_projections,
            _,
            range_image_top_pose,
        ) = parse_range_image_and_camera_projection(frame)

        cam_calibs = {calib.name: calib for calib in frame.context.camera_calibrations}
        images = {img.name: img for img in frame.images}

        # Camera-only split may not include LiDAR points. Save zero depth maps to keep layout consistent.
        if range_image_top_pose is None:
            for cam_name, _ in images.items():
                cam_calib = cam_calibs[cam_name]
                h, w = cam_calib.height, cam_calib.width
                depth_map = np.zeros((h, w), dtype=np.float32)
                depth_map_save_path = (
                    f"{self.save_dir}/{str(file_idx).zfill(3)}/depth_map/"
                    f"{str(frame_idx).zfill(3)}_{str(cam_name - 1)}.npz"
                )
                np.savez_compressed(
                    depth_map_save_path,
                    depth=depth_map,
                    cam_name=np.int32(cam_name),
                    cam_id=np.int32(cam_name - 1),
                    timestamp=np.int64(frame.timestamp_micros),
                )
            return

        range_images_flow, _, _ = parse_range_image_flow_and_camera_projection(frame)
        (
            _,
            points,
            _,
            _,
            _,
            _,
            _,
        ) = convert_range_image_to_point_cloud_flow(
            frame,
            range_images,
            range_images_flow,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
        )

        # LiDAR points are in Waymo vehicle frame (ego frame).
        lidar_points = np.concatenate(points, axis=0).astype(np.float32)
        lidar_points_homo = np.concatenate(
            [lidar_points, np.ones((lidar_points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )

        for cam_name, _ in images.items():
            cam_calib = cam_calibs[cam_name]
            h, w = cam_calib.height, cam_calib.width
            depth_map = np.zeros((h, w), dtype=np.float32)

            cam_to_ego = np.array(cam_calib.extrinsic.transform, dtype=np.float32).reshape(4, 4)
            # Keep the same camera-coordinate convention as waymo_sourceloader.
            cam_to_ego = cam_to_ego @ OPENCV2DATASET
            ego_to_cam = np.linalg.inv(cam_to_ego)

            cam_lidar_points = (ego_to_cam @ lidar_points_homo.T).T
            depth_mask = cam_lidar_points[:, 2] > 0
            cam_lidar_points = cam_lidar_points[depth_mask]

            if cam_lidar_points.shape[0] == 0:
                depth_map_save_path = (
                    f"{self.save_dir}/{str(file_idx).zfill(3)}/depth_map/"
                    f"{str(frame_idx).zfill(3)}_{str(cam_name - 1)}.npz"
                )
                np.savez_compressed(
                    depth_map_save_path,
                    depth=depth_map,
                    cam_name=np.int32(cam_name),
                    cam_id=np.int32(cam_name - 1),
                    timestamp=np.int64(frame.timestamp_micros),
                )
                continue

            intrinsics = np.array(
                [
                    [cam_calib.intrinsic[0], 0.0, cam_calib.intrinsic[2]],
                    [0.0, cam_calib.intrinsic[1], cam_calib.intrinsic[3]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            pixel_points = (intrinsics @ cam_lidar_points[:, :3].T).T
            pixel_points[:, :2] /= np.maximum(pixel_points[:, 2:3], 1e-6)

            pixel_mask = (
                (pixel_points[:, 0] >= 0)
                & (pixel_points[:, 0] <= w - 1)
                & (pixel_points[:, 1] >= 0)
                & (pixel_points[:, 1] <= h - 1)
            )
            valid_points = np.round(pixel_points[pixel_mask, :2]).astype(np.int32)
            valid_depth = cam_lidar_points[pixel_mask, 2]

            if valid_points.shape[0] > 0:
                u = valid_points[:, 0]
                v = valid_points[:, 1]
                sort_idx = np.argsort(valid_depth)
                u = u[sort_idx]
                v = v[sort_idx]
                d = valid_depth[sort_idx]

                flat = v * w + u
                _, unique_idx = np.unique(flat, return_index=True)
                depth_map[v[unique_idx], u[unique_idx]] = d[unique_idx]

            depth_map_save_path = (
                f"{self.save_dir}/{str(file_idx).zfill(3)}/depth_map/"
                f"{str(frame_idx).zfill(3)}_{str(cam_name - 1)}.npz"
            )
            np.savez_compressed(
                depth_map_save_path,
                depth=depth_map,
                cam_name=np.int32(cam_name),
                cam_id=np.int32(cam_name - 1),
                timestamp=np.int64(frame.timestamp_micros),
            )

    def save_dynamic_mask(self, frame, file_idx, frame_idx, class_valid='all'):
        assert class_valid in ['all', 'human', 'vehicle'], "Invalid class valid"
        if class_valid == 'all':
            VALID_CLASSES = WAYMO_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = WAYMO_HUMAN_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = WAYMO_VEHICLE_CLASSES
        mask_dir = f"{self.save_dir}/{str(file_idx).zfill(3)}/dynamic_masks/{class_valid}"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
            
        """Parse and save the segmentation data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            # dynamic_mask
            img_path = (
                f"{self.save_dir}/{str(file_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.jpg"
            )
            img_shape = np.array(Image.open(img_path))
            dynamic_mask = np.zeros_like(img_shape, dtype=np.float32)[..., 0]

            filter_available = any(
                [label.num_top_lidar_points_in_box > 0 for label in frame.laser_labels]
            )
            calibration = next(
                cc for cc in frame.context.camera_calibrations if cc.name == img.name
            )
            for label in frame.laser_labels:
                # camera_synced_box is not available for the data with flow.
                # box = label.camera_synced_box
                
                class_name = WAYMO_CLASSES[label.type]
                if class_name not in VALID_CLASSES:
                    continue

                box = label.box
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])
                if not box.ByteSize():
                    continue  # Filter out labels that do not have a camera_synced_box.
                if (filter_available and not label.num_top_lidar_points_in_box) or (
                    not filter_available and not label.num_lidar_points_in_box
                ):
                    continue  # Filter out likely occluded objects.

                # Retrieve upright 3D box corners.
                box_coords = np.array(
                    [
                        [
                            box.center_x,
                            box.center_y,
                            box.center_z,
                            box.length,
                            box.width,
                            box.height,
                            box.heading,
                        ]
                    ]
                )
                corners = box_utils.get_upright_3d_box_corners(box_coords)[
                    0
                ].numpy()  # [8, 3]

                # Project box corners from vehicle coordinates onto the image.
                projected_corners = project_vehicle_to_image(
                    frame.pose, calibration, corners
                )
                u, v, ok = projected_corners.transpose()
                ok = ok.astype(bool)

                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not all(ok):
                    continue
                u = u[ok]
                v = v[ok]

                # Clip box to image bounds.
                u = np.clip(u, 0, calibration.width)
                v = np.clip(v, 0, calibration.height)

                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue

                # Draw projected 2D box onto the image.
                xy = (u.min(), v.min())
                width = u.max() - u.min()
                height = v.max() - v.min()
                # max pooling
                dynamic_mask[
                    int(xy[1]) : int(xy[1] + height),
                    int(xy[0]) : int(xy[0] + width),
                ] = np.maximum(
                    dynamic_mask[
                        int(xy[1]) : int(xy[1] + height),
                        int(xy[0]) : int(xy[0] + width),
                    ],
                    speed,
                )
            # thresholding, use 1.0 m/s to determine whether the pixel is moving
            dynamic_mask = np.clip((dynamic_mask > 1.0) * 255, 0, 255).astype(np.uint8)
            dynamic_mask = Image.fromarray(dynamic_mask, "L")
            dynamic_mask_path = os.path.join(mask_dir, f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.png")
            dynamic_mask.save(dynamic_mask_path)
            
    def save_objects(self, dataset):
        """Parse and save the ground truth bounding boxes."""
        instances_info, frame_instances = {}, {}
        
        for frame_idx, data in enumerate(dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            frame_instances[frame_idx] = []
            for l in frame.laser_labels:
                frame_pose = np.array(frame.pose.transform).reshape(4, 4)
                
                str_id = str(l.id)
                if WAYMO_CLASSES[l.type] not in WAYMO_DYNAMIC_CLASSES:
                    continue
                
                frame_instances[frame_idx].append(str_id)
                
                if str_id not in instances_info:
                    instances_info[str_id] = dict(
                        id=l.id,
                        # class_ind=l.type,
                        class_name=WAYMO_CLASSES[l.type],
                        frame_annotations={
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        }
                    )
                
                # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
                box = l.box
                
                # Box coordinates in vehicle frame.
                tx, ty, tz = box.center_x, box.center_y, box.center_z
                
                # The heading of the bounding box (in radians).  The heading is the angle
                #   required to rotate +x to the surface normal of the box front face. It is
                #   normalized to [-pi, pi).
                c = np.math.cos(box.heading)
                s = np.math.sin(box.heading)
                
                # [object to vehicle]
                # https://github.com/gdlg/simple-waymo-open-dataset-reader/blob/d488196b3ded6574c32fad391467863b948dfd8e/simple_waymo_open_dataset_reader/utils.py#L32
                o2v = np.array([
                    [ c, -s,  0, tx],
                    [ s,  c,  0, ty],
                    [ 0,  0,  1, tz],
                    [ 0,  0,  0,  1]])
                
                # [object to ENU world]
                pose = frame_pose @ o2v # o2w = v2w @ o2v
                
                # difficulty = l.detection_difficulty_level
                
                # tracking_difficulty = l.tracking_difficulty_level
                
                # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                # length: dim_x: along heading; dim_y: verticle to heading; dim_z: verticle up
                dimension = [box.length, box.width, box.height]
                
                instances_info[str_id]['frame_annotations']['frame_idx'].append(frame_idx)
                instances_info[str_id]['frame_annotations']['obj_to_world'].append(pose.tolist())
                instances_info[str_id]['frame_annotations']['box_size'].append(dimension)
                
        # Correct ID mapping
        id_map = {}
        for i, (k, v) in enumerate(instances_info.items()):
            id_map[v["id"]] = i

        # Update keys in instances_info
        new_instances_info = {}
        for k, v in instances_info.items():
            new_instances_info[id_map[v["id"]]] = v

        # Update keys in frame_instances
        new_frame_instances = {}
        for k, v in frame_instances.items():
            new_frame_instances[k] = [id_map[i] for i in v]

        return new_instances_info, new_frame_instances

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for i in id_list:
            if "images" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/images", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/sky_masks", exist_ok=True)
            if "calib" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/extrinsics", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/intrinsics", exist_ok=True)
            if "pose" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/ego_pose", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks", exist_ok=True)
            if "depth_map" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/depth_map", exist_ok=True)
            if "objects" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances", exist_ok=True)
