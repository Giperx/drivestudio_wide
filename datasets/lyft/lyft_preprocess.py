"""
Lyft数据集预处理脚本

基于lyft_dataset_sdk，参考nuscenes_preprocess.py结构

Lyft数据集特点：
- 标注频率: 5Hz (固定，无需插值)
- JSON格式与nuScenes相同
- 坐标系与nuScenes相同 (OpenCV相机坐标系，FLU自车坐标系)
- 7个相机 (多了CAM_FRONT_ZOOMED)
- 1或3个LiDAR (取决于车辆)

使用方法:
export PYTHONPATH=$(pwd)
# 处理1224分辨率训练集
python datasets/preprocess.py \
    --data_root data/raw/lyft_train/v1.01-train \
    --dataset lyft \
    --split_file third_party/futr3d/data/lyft/train1224.txt \
    --target_dir data/lyft/processed \
    --workers 1 \
    --process_keys cam2ego_extrinsics ego_pose depth_map images lidar calib objects dynamic_masks

# 处理1920分辨率训练集
python datasets/preprocess.py \
    --data_root data/raw/lyft_train/v1.01-train \
    --dataset lyft \
    --split_file data/lyft_val1920.txt \
    --target_dir data/lyft/processed \
    --workers 1 \
    --process_keys cam2ego_extrinsics ego_pose depth_map images lidar calib objects dynamic_masks
    
lyft_train1224 lyft_train1920 lyft_val1224 lyft_val1920

conda activate segformer
CUDA_VISIBLE_DEVICES=1 python datasets/tools/extract_masks.py \
    --data_root data/lyft/processed/lyft_val1920 \
    --segformer_path=./SegFormer \
    --checkpoint=./SegFormer/segformer.b5.1024x1024.city.160k.pth \
    --start_idx 0 \
    --num_scenes 5 \
    --process_dynamic_mask
"""

import json
import os
from collections import Counter
from typing import List

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from lyft_dataset_sdk.lyftdataset import LyftDataset
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

from datasets.tools.multiprocess_utils import track_parallel_progress

# Lyft的9个物体类别
LYFT_CATEGORIES = [
    'car',
    'other_vehicle',
    'pedestrian',
    'bicycle',
    'truck',
    'bus',
    'motorcycle',
    'animal',
    'emergency_vehicle'
]

# 静态属性 (有这些属性的物体应该排除在动态mask之外)
LYFT_STATIC_ATTRIBUTES = [
    'is_stationary',
    'object_action_parked',
]

# 动态属性 (用于判断物体是否动态)
LYFT_DYNAMIC_ATTRIBUTES = [
    'object_action_driving_straight_forward',
    'object_action_walking',
    'object_action_right_turn',
    'object_action_left_turn',
    'object_action_lane_change_left',
    'object_action_lane_change_right',
    'object_action_running',
    'object_action_standing',
    'object_action_sitting',
    'object_action_u_turn',
    'object_action_reversing',
    'object_action_gliding_on_wheels',
    'object_action_abnormal_or_traffic_violation',
    'object_action_loss_of_control',
    'object_action_other_motion'
]

# human类别 (行人, 自行车, 摩托车等非刚体)
LYFT_HUMAN_CLASSES = [
    'pedestrian',
    'bicycle',
    'motorcycle'
]

# vehicle类别 (车辆类 - 刚体)
LYFT_VEHICLE_CLASSES = [
    'car',
    'truck',
    'bus',
    'other_vehicle',
    'emergency_vehicle',
]

# other_dynamics类别 (除人以外的非刚体: 动物等)
LYFT_OTHER_DYNAMIC_CLASSES = [
    'animal'
]

# 所有动态类别
LYFT_DYNAMIC_CLASSES = LYFT_HUMAN_CLASSES + LYFT_VEHICLE_CLASSES + LYFT_OTHER_DYNAMIC_CLASSES

# 兼容旧代码的别名
LYFT_NONRIGID_DYNAMIC_CLASSES = LYFT_HUMAN_CLASSES + LYFT_OTHER_DYNAMIC_CLASSES
LYFT_RIGID_DYNAMIC_CLASSES = LYFT_VEHICLE_CLASSES


class LyftProcessor(object):
    """Process Lyft dataset.

    Lyft数据集特点:
    - 标注频率: 5Hz (固定)
    - 相机: 7个 (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
               CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_BACK, CAM_FRONT_ZOOMED)
    - LiDAR: 1个 (LIDAR_TOP) 或 3个 (加上 LIDAR_FRONT_LEFT, LIDAR_FRONT_RIGHT)
    - 坐标系: 与nuScenes相同 (OpenCV相机坐标系)

    Args:
        load_dir (str): Source directory of Lyft data (v1.01-train).
        save_dir (str): Target directory for processed data.
        split_name (str): Name of the split (e.g., train1224, val1920).
        scene_names (list): List of scene names to process.
        workers (int): Number of parallel processing workers.
        process_keys (list): Data types to process.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        split_name='train1224',
        scene_names=None,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "cam2ego_extrinsics",
            "ego_pose",
            "depth_map",
            "objects"
        ],
        workers=64,
    ):
        self.scene_names = scene_names
        self.process_keys = process_keys
        print("Will process keys: ", self.process_keys)

        self.load_dir = load_dir
        self.save_dir = os.path.join(save_dir, split_name)
        self.workers = int(workers)

        # 初始化Lyft数据集
        json_path = os.path.join(load_dir, 'v1.01-train')
        self.lyft = LyftDataset(
            data_path=load_dir,
            json_path=json_path,
            verbose=True
        )

        # 标准6个相机 (与nuScenes一致)
        self.cam_list = [
            "CAM_FRONT",        # 0
            "CAM_FRONT_LEFT",   # 1
            "CAM_FRONT_RIGHT",  # 2
            "CAM_BACK_LEFT",    # 3
            "CAM_BACK_RIGHT",   # 4
            "CAM_BACK"          # 5
        ]

        # LiDAR
        self.lidar_list = ['LIDAR_TOP']

        # 构建scene name到scene index的映射
        self.scene_name_to_idx = {}
        for i, scene in enumerate(self.lyft.scene):
            self.scene_name_to_idx[scene['name']] = i

        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.scene_names is None:
            # 处理所有场景
            scene_indices = range(len(self.lyft.scene))
        else:
            # 根据scene name获取scene index
            scene_indices = []
            for name in self.scene_names:
                if name in self.scene_name_to_idx:
                    scene_indices.append(self.scene_name_to_idx[name])
                else:
                    print(f"Warning: Scene {name} not found in dataset")

        track_parallel_progress(self.convert_one, scene_indices, self.workers)
        print("\nFinished ...")

    def convert_one(self, scene_idx):
        """Convert action for single scene."""
        scene = self.lyft.scene[scene_idx]
        scene_name = scene['name']

        # 找到该scene在输出目录中的编号
        if self.scene_names is not None:
            try:
                output_idx = self.scene_names.index(scene_name)
            except ValueError:
                print(f"Warning: Scene {scene_name} not in scene_names list")
                return
        else:
            output_idx = scene_idx

        if "images" in self.process_keys:
            self.save_image(scene, output_idx)
            print(f"Processed images for scene {str(output_idx).zfill(3)} ({scene_name})")
        if "calib" in self.process_keys:
            self.save_calib(scene, output_idx)
            print(f"Processed calib for scene {str(output_idx).zfill(3)}")
        if "cam2ego_extrinsics" in self.process_keys:
            self.save_cam2ego_extrinsics(scene, output_idx)
            print(f"Processed cam2ego extrinsics for scene {str(output_idx).zfill(3)}")
        if "lidar" in self.process_keys:
            self.save_lidar(scene, output_idx)
            print(f"Processed lidar for scene {str(output_idx).zfill(3)}")
        if "ego_pose" in self.process_keys:
            self.save_ego_pose(scene, output_idx)
            print(f"Processed ego poses for scene {str(output_idx).zfill(3)}")
        if "depth_map" in self.process_keys:
            self.save_depth_map(scene, output_idx)
            print(f"Processed depth maps for scene {str(output_idx).zfill(3)}")
        if "objects" in self.process_keys:
            self.save_objects(scene, output_idx)
            print(f"Processed objects for scene {str(output_idx).zfill(3)}")
        if "dynamic_masks" in self.process_keys:
            self.save_dynamic_mask(scene, output_idx, class_valid='all')
            self.save_dynamic_mask(scene, output_idx, class_valid='human')
            self.save_dynamic_mask(scene, output_idx, class_valid='vehicle')
            self.save_dynamic_mask(scene, output_idx, class_valid='other_dynamics')
            print(f"Processed dynamic masks for scene {str(output_idx).zfill(3)}")

    def get_keyframe_samples(self, scene):
        """获取场景的所有关键帧sample"""
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']

        samples = []
        current_sample = self.lyft.get('sample', first_sample_token)

        while True:
            samples.append(current_sample)
            if current_sample['token'] == last_sample_token:
                break
            current_sample = self.lyft.get('sample', current_sample['next'])

        return samples

    def save_image(self, scene, scene_idx):
        """Parse and save the images in jpg format."""
        samples = self.get_keyframe_samples(scene)

        for frame_idx, sample in enumerate(samples):
            for cam_idx, cam_name in enumerate(self.cam_list):
                cam_data = self.lyft.get('sample_data', sample['data'][cam_name])
                source_path = os.path.join(self.lyft.data_path, cam_data['filename'])

                img_path = (
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                    + f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg"
                )

                # 复制图像
                os.system(f"cp {source_path} {img_path}")

    def save_calib(self, scene, scene_idx):
        """Parse and save the calibration data (intrinsics + cam2world)."""
        samples = self.get_keyframe_samples(scene)

        for frame_idx, sample in enumerate(samples):
            for cam_idx, cam_name in enumerate(self.cam_list):
                cam_data = self.lyft.get('sample_data', sample['data'][cam_name])
                calib_data = self.lyft.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

                # Extrinsics (camera to ego)
                extrinsics_cam_to_ego = np.eye(4)
                extrinsics_cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
                extrinsics_cam_to_ego[:3, 3] = np.array(calib_data['translation'])

                # Get ego pose (ego to world)
                ego_pose_data = self.lyft.get('ego_pose', cam_data['ego_pose_token'])
                ego_to_world = np.eye(4)
                ego_to_world[:3, :3] = Quaternion(ego_pose_data['rotation']).rotation_matrix
                ego_to_world[:3, 3] = np.array(ego_pose_data['translation'])

                # Transform camera extrinsics to world coordinates
                extrinsics_cam_to_world = ego_to_world @ extrinsics_cam_to_ego

                np.savetxt(
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/extrinsics/"
                    f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.txt",
                    extrinsics_cam_to_world
                )

                # Intrinsics
                intrinsics = np.array(calib_data['camera_intrinsic'])
                # fx fy cx cy 0 0 0 0 0
                Ks = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2],
                      0., 0., 0., 0., 0.]
                np.savetxt(
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/intrinsics/"
                    f"{str(cam_idx)}.txt",
                    Ks
                )

    def save_cam2ego_extrinsics(self, scene, scene_idx):
        """Save per-camera cam-to-ego extrinsics once per scene (static)."""
        first_sample = self.lyft.get('sample', scene['first_sample_token'])

        for cam_idx, cam_name in enumerate(self.cam_list):
            cam_data = self.lyft.get('sample_data', first_sample['data'][cam_name])
            calib_data = self.lyft.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

            cam_to_ego = np.eye(4)
            cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
            cam_to_ego[:3, 3] = np.array(calib_data['translation'])

            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/cam2ego_extrinsics/"
                f"{str(cam_idx)}.txt",
                cam_to_ego
            )

    def save_ego_pose(self, scene, scene_idx):
        """Save per-frame ego-to-world pose matrices."""
        samples = self.get_keyframe_samples(scene)
        front_cam = self.cam_list[0]

        for frame_idx, sample in enumerate(samples):
            cam_data = self.lyft.get('sample_data', sample['data'][front_cam])
            ego_pose_data = self.lyft.get('ego_pose', cam_data['ego_pose_token'])

            ego_to_world = np.eye(4)
            ego_to_world[:3, :3] = Quaternion(ego_pose_data['rotation']).rotation_matrix
            ego_to_world[:3, 3] = np.array(ego_pose_data['translation'])

            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/ego_pose/"
                f"{str(frame_idx).zfill(3)}.txt",
                ego_to_world
            )

    def save_lidar(self, scene, scene_idx):
        """Parse and save the lidar data in bin format and lidar pose."""
        samples = self.get_keyframe_samples(scene)

        for frame_idx, sample in enumerate(samples):
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.lyft.get('sample_data', lidar_token)
            lidar_path = os.path.join(self.lyft.data_path, lidar_data['filename'])

            # 读取点云 (x, y, z, intensity, ring_id)
            point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

            # 保存点云 (只保留xyz)
            lidar_save_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar/{str(frame_idx).zfill(3)}.bin"
            point_cloud[:, :3].astype(np.float32).tofile(lidar_save_path)

            # 获取lidar extrinsics (lidar to ego)
            calib_data = self.lyft.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_to_ego = np.eye(4)
            lidar_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
            lidar_to_ego[:3, 3] = np.array(calib_data['translation'])

            # 获取ego pose (ego to world)
            ego_pose_data = self.lyft.get('ego_pose', lidar_data['ego_pose_token'])
            ego_to_world = np.eye(4)
            ego_to_world[:3, :3] = Quaternion(ego_pose_data['rotation']).rotation_matrix
            ego_to_world[:3, 3] = np.array(ego_pose_data['translation'])

            # 计算lidar在世界坐标系中的pose
            lidar_to_world = ego_to_world @ lidar_to_ego

            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar_pose/"
                f"{str(frame_idx).zfill(3)}.txt",
                lidar_to_world
            )

    def save_depth_map(self, scene, scene_idx):
        """Save per-frame camera depth maps from LiDAR projection."""
        samples = self.get_keyframe_samples(scene)

        for frame_idx, sample in enumerate(samples):
            # 获取LiDAR点云
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.lyft.get('sample_data', lidar_token)
            lidar_path = os.path.join(self.lyft.data_path, lidar_data['filename'])
            point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            lidar_points = point_cloud[:, :3]

            # LiDAR to ego
            lidar_calib = self.lyft.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_to_ego = np.eye(4)
            lidar_to_ego[:3, :3] = Quaternion(lidar_calib['rotation']).rotation_matrix
            lidar_to_ego[:3, 3] = np.array(lidar_calib['translation'])

            # Ego to world
            lidar_ego_pose = self.lyft.get('ego_pose', lidar_data['ego_pose_token'])
            ego_to_world = np.eye(4)
            ego_to_world[:3, :3] = Quaternion(lidar_ego_pose['rotation']).rotation_matrix
            ego_to_world[:3, 3] = np.array(lidar_ego_pose['translation'])

            # LiDAR to world
            lidar_to_world = ego_to_world @ lidar_to_ego

            # 转换点云到世界坐标系
            lidar_points_homo = np.concatenate(
                [lidar_points, np.ones((lidar_points.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            points_world = (lidar_to_world @ lidar_points_homo.T).T

            # 为每个相机生成深度图
            for cam_idx, cam_name in enumerate(self.cam_list):
                cam_data = self.lyft.get('sample_data', sample['data'][cam_name])
                cam_calib = self.lyft.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

                # Camera to ego
                sensor_to_ego = np.eye(4)
                sensor_to_ego[:3, :3] = Quaternion(cam_calib['rotation']).rotation_matrix
                sensor_to_ego[:3, 3] = np.array(cam_calib['translation'])
                ego_to_sensor = np.linalg.inv(sensor_to_ego)

                # Ego to world (at camera timestamp)
                cam_ego_pose = self.lyft.get('ego_pose', cam_data['ego_pose_token'])
                world_to_ego = np.eye(4)
                world_to_ego[:3, :3] = Quaternion(cam_ego_pose['rotation']).inverse.rotation_matrix
                world_to_ego[:3, 3] = -world_to_ego[:3, :3] @ np.array(cam_ego_pose['translation'])

                # World to camera
                world_to_sensor = ego_to_sensor @ world_to_ego
                points_camera = (world_to_sensor @ points_world.T).T

                # 过滤掉相机后面的点
                depth_mask = points_camera[:, 2] > 0
                points_camera = points_camera[depth_mask]

                h = cam_data['height']
                w = cam_data['width']
                depth_map = np.zeros((h, w), dtype=np.float32)

                if points_camera.shape[0] == 0:
                    depth_map_save_path = (
                        f"{self.save_dir}/{str(scene_idx).zfill(3)}/depth_map/"
                        f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.npz"
                    )
                    np.savez_compressed(
                        depth_map_save_path,
                        depth=depth_map,
                        cam_name=cam_name,
                        cam_id=np.int32(cam_idx),
                        timestamp=np.int64(sample['timestamp']),
                    )
                    continue

                # 投影到图像平面
                intrinsics = np.array(cam_calib['camera_intrinsic'], dtype=np.float32)
                pixel_points = (intrinsics @ points_camera[:, :3].T).T
                pixel_points[:, :2] /= np.maximum(pixel_points[:, 2:3], 1e-6)

                # 过滤图像外的点
                pixel_mask = (
                    (pixel_points[:, 0] >= 0)
                    & (pixel_points[:, 0] <= w - 1)
                    & (pixel_points[:, 1] >= 0)
                    & (pixel_points[:, 1] <= h - 1)
                )
                valid_points = np.round(pixel_points[pixel_mask, :2]).astype(np.int32)
                valid_depth = points_camera[pixel_mask, 2]

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
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/depth_map/"
                    f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.npz"
                )
                np.savez_compressed(
                    depth_map_save_path,
                    depth=depth_map,
                    cam_name=cam_name,
                    cam_id=np.int32(cam_idx),
                    timestamp=np.int64(sample['timestamp']),
                )

    def save_objects(self, scene, scene_idx):
        """Parse and save the objects annotation data."""
        samples = self.get_keyframe_samples(scene)

        instances_info = {}
        frame_instances = {}

        for frame_idx, sample in enumerate(samples):
            # 获取该帧的所有标注
            annotations = [self.lyft.get('sample_annotation', token)
                          for token in sample['anns']]

            frame_instances[frame_idx] = []

            for ann in annotations:
                # 获取类别 (Lyft直接包含category_name)
                category_name = ann['category_name']

                # 获取属性判断动态/静态
                is_dynamic = False
                for attr_token in ann.get('attribute_tokens', []):
                    attr = self.lyft.get('attribute', attr_token)
                    if attr['name'] in LYFT_DYNAMIC_ATTRIBUTES:
                        is_dynamic = True
                        break

                instance_token = ann['instance_token']

                if instance_token not in instances_info:
                    instances_info[instance_token] = {
                        'id': instance_token,
                        'class_name': category_name,
                        'is_dynamic': is_dynamic,
                        'frame_annotations': {
                            'frame_idx': [],
                            'obj_to_world': [],
                            'box_size': [],
                        }
                    }

                # Object to world transformation
                o2w = np.eye(4)
                o2w[:3, :3] = Quaternion(ann['rotation']).rotation_matrix
                o2w[:3, 3] = np.array(ann['translation'])

                # Box size (wlh -> lwh)
                # nuScenes/Lyft使用 (width, length, height)
                # 我们转换为 (length, width, height)
                size_wlh = ann['size']
                size_lwh = [size_wlh[1], size_wlh[0], size_wlh[2]]

                instances_info[instance_token]['frame_annotations']['frame_idx'].append(frame_idx)
                instances_info[instance_token]['frame_annotations']['obj_to_world'].append(o2w.tolist())
                instances_info[instance_token]['frame_annotations']['box_size'].append(size_lwh)

                frame_instances[frame_idx].append(instance_token)

        # Correct ID mapping
        id_map = {}
        for i, (k, v) in enumerate(instances_info.items()):
            id_map[v["id"]] = i

        new_instances_info = {}
        for k, v in instances_info.items():
            new_instances_info[id_map[v["id"]]] = v

        new_frame_instances = {}
        for k, v in frame_instances.items():
            new_frame_instances[k] = [id_map[i] for i in v]

        # Save
        instances_info_save_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances"
        with open(f"{instances_info_save_path}/instances_info.json", "w") as fp:
            json.dump(new_instances_info, fp, indent=4)
        with open(f"{instances_info_save_path}/frame_instances.json", "w") as fp:
            json.dump(new_frame_instances, fp, indent=4)

    def save_dynamic_mask(self, scene, scene_idx, class_valid='all'):
        """Parse and save the dynamic mask data.

        只处理动态物体，排除有is_stationary或object_action_parked属性的物体。

        Args:
            scene: scene对象
            scene_idx: 场景输出索引
            class_valid: 'all', 'human', 'vehicle', 或 'other_dynamics'
                - 'all': 所有动态类别
                - 'human': 只有pedestrian
                - 'vehicle': 车辆类 (car, truck, bus, other_vehicle, emergency_vehicle)
                - 'other_dynamics': 除人以外的非刚体 (bicycle, motorcycle, animal)
        """
        assert class_valid in ['all', 'human', 'vehicle', 'other_dynamics'], "Invalid class_valid"

        if class_valid == 'all':
            VALID_CLASSES = LYFT_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = LYFT_HUMAN_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = LYFT_VEHICLE_CLASSES
        elif class_valid == 'other_dynamics':
            VALID_CLASSES = LYFT_OTHER_DYNAMIC_CLASSES

        mask_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/dynamic_masks/{class_valid}"
        os.makedirs(mask_dir, exist_ok=True)

        samples = self.get_keyframe_samples(scene)

        for frame_idx, sample in enumerate(samples):
            for cam_idx, cam_name in enumerate(self.cam_list):
                cam_data = self.lyft.get('sample_data', sample['data'][cam_name])
                img_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg"

                img = cv2.imread(img_path)
                if img is None:
                    continue
                dynamic_mask = np.zeros(img.shape[:2], dtype=np.float32)

                # 获取该帧的所有标注
                anns = [self.lyft.get('sample_annotation', token) for token in sample['anns']]

                # 筛选有效的动态标注
                valid_anns = []
                for ann in anns:
                    # 首先检查类别是否在目标类别中
                    if ann['category_name'] not in VALID_CLASSES:
                        continue

                    # 检查属性，排除静态物体
                    is_static = False
                    for attr_token in ann.get('attribute_tokens', []):
                        attr = self.lyft.get('attribute', attr_token)
                        if attr['name'] in LYFT_STATIC_ATTRIBUTES:
                            is_static = True
                            break

                    # 只有非静态物体才加入有效标注列表
                    if not is_static:
                        valid_anns.append(ann)

                # 获取相机内参
                cs_record = self.lyft.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                camera_intrinsic = np.array(cs_record['camera_intrinsic'])

                # 获取ego pose
                pose_record = self.lyft.get('ego_pose', cam_data['ego_pose_token'])

                # 将3D框投影到2D并生成mask
                for ann in valid_anns:
                    # 创建Box对象 (translation, size, rotation)
                    # Lyft的size是 [width, length, height]
                    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

                    # 转换到ego坐标系
                    box.translate(-np.array(pose_record['translation']))
                    box.rotate(Quaternion(pose_record['rotation']).inverse)

                    # 转换到相机坐标系
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)

                    # 投影3D框到2D
                    corners_3d = box.corners()
                    corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)

                    # 检查是否在相机前方且在图像内
                    in_front = np.all(corners_3d[2, :] > 0.1)
                    in_image = (np.all(corners_2d[0, :] >= 0) &
                               np.all(corners_2d[0, :] < img.shape[1]) &
                               np.all(corners_2d[1, :] >= 0) &
                               np.all(corners_2d[1, :] < img.shape[0]))
                    if not (in_front and in_image):
                        continue

                    # 提取2D坐标
                    corners_2d = corners_2d[:2, :]

                    # 填充mask
                    u = corners_2d[0, :].astype(np.int32)
                    v = corners_2d[1, :].astype(np.int32)
                    u = np.clip(u, 0, img.shape[1] - 1)
                    v = np.clip(v, 0, img.shape[0] - 1)

                    if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                        continue

                    xy = (u.min(), v.min())
                    width = u.max() - u.min()
                    height = v.max() - v.min()

                    dynamic_mask[
                        int(xy[1]): int(xy[1] + height),
                        int(xy[0]): int(xy[0] + width)
                    ] = np.maximum(
                        dynamic_mask[
                            int(xy[1]): int(xy[1] + height),
                            int(xy[0]): int(xy[0] + width)
                        ],
                        1
                    )

                # 保存动态mask
                dynamic_mask = np.clip((dynamic_mask > 0.) * 255, 0, 255).astype(np.uint8)
                dynamic_mask = Image.fromarray(dynamic_mask, "L")
                dynamic_mask_path = f"{mask_dir}/{str(frame_idx).zfill(3)}_{str(cam_idx)}.png"
                dynamic_mask.save(dynamic_mask_path)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.scene_names is not None:
            id_list = range(len(self.scene_names))
        else:
            id_list = range(len(self.lyft.scene))

        for i in id_list:
            if "images" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/images", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/sky_masks", exist_ok=True)
            if "calib" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/extrinsics", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/intrinsics", exist_ok=True)
            if "cam2ego_extrinsics" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/cam2ego_extrinsics", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar_pose", exist_ok=True)
            if "ego_pose" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/ego_pose", exist_ok=True)
            if "depth_map" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/depth_map", exist_ok=True)
            if "objects" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks/all", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks/human", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks/vehicle", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks/other_dynamics", exist_ok=True)
