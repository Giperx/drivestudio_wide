"""Preprocess CARLA syn-nuScenes scenes into the DriveStudio nuScenes layout.

Each scene directory is its own nuScenes dataroot (samples/, sweeps/, v1.0-*).
Capture is already 10 Hz, including non-keyframes, so this processor walks the
real sample_data chain and does not interpolate. There are no 3D boxes, so
dynamic masks and objects are skipped.

Camera ids follow the nuScenes rear-camera slots:

    5  CAM_BACK_SOURCE
    4  CAM_BACK_RIGHT_SOURCE
    3  CAM_BACK_LEFT_SOURCE
    2  CAM_BACK_WIDE_RECT_GT   (images and calib only, no depth map)

CAM_BACK_WIDE_CYL_GT is cylindrical and is not exported.
"""

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from datasets.tools.multiprocess_utils import track_parallel_progress


# Tables the devkit always loads. Syn scenes omit annotation and map tables.
_STUB_TABLES = (
    "category",
    "attribute",
    "visibility",
    "instance",
    "sample_annotation",
    "map",
)
_SKIP_KEYS = ("dynamic_masks", "objects", "objects_vis")


class SynNuScenesProcessor(object):
    """Export one or more syn-nuScenes scenes at native 10 Hz."""

    # Explicit ids, not list order. Do not enumerate() these.
    CAMERAS = {
        5: "CAM_BACK_SOURCE",
        4: "CAM_BACK_RIGHT_SOURCE",
        3: "CAM_BACK_LEFT_SOURCE",
        2: "CAM_BACK_WIDE_RECT_GT",
    }
    CAM_IDS = (5, 4, 3, 2)
    # Wide rectified GT is kept as an image, but lidar depth is not rasterized.
    SKIP_DEPTH_CAM_IDS = {2}
    LIDAR_CHANNEL = "LIDAR_TOP"
    POSE_CAM_ID = 5

    def __init__(
        self,
        load_dir,
        save_dir,
        split="v1.0-trainval",
        process_keys=None,
        process_id_list=None,
        workers=1,
    ):
        self.load_dir = load_dir
        self.split = split
        self.scene_dirs = self._discover_scenes(load_dir, split)
        self.process_id_list = (
            list(range(len(self.scene_dirs)))
            if process_id_list is None
            else [int(i) for i in process_id_list]
        )
        for scene_idx in self.process_id_list:
            if scene_idx < 0 or scene_idx >= len(self.scene_dirs):
                raise IndexError(
                    f"scene index {scene_idx} out of range for {len(self.scene_dirs)} "
                    f"syn-nuScenes scenes under {load_dir}"
                )

        requested = list(process_keys) if process_keys is not None else [
            "images",
            "lidar",
            "calib",
            "cam2ego_extrinsics",
            "ego_pose",
            "depth_map",
        ]
        skipped = [key for key in requested if key in _SKIP_KEYS]
        if skipped:
            print(
                "syn_nuscenes has no dynamic objects or 3D boxes; "
                f"skipping {skipped}"
            )
        self.process_keys = [key for key in requested if key not in _SKIP_KEYS]
        print("will process keys: ", self.process_keys)
        print(
            "cameras: "
            + ", ".join(f"{cam_id}:{self.CAMERAS[cam_id]}" for cam_id in self.CAM_IDS)
        )
        print(
            "depth maps skipped for camera ids: "
            + ", ".join(str(cam_id) for cam_id in sorted(self.SKIP_DEPTH_CAM_IDS))
        )
        print("native 10 Hz, no interpolation")
        for scene_idx in self.process_id_list:
            print(
                f"scene {scene_idx:03d} <- {os.path.basename(self.scene_dirs[scene_idx])}"
            )

        # Match the nuScenes 10 Hz layout: <target>_10Hz/<split tail>/<scene_idx>
        if "processed_10Hz" not in save_dir:
            save_dir = save_dir.replace("processed", "processed_10Hz")
        self.save_dir = os.path.join(save_dir, split.split("-")[-1])
        self.workers = int(workers)
        self.create_folder()

    @staticmethod
    def _discover_scenes(load_dir, split):
        """Return scene dataroots. A dataroot holds <split>/scene.json."""
        direct = os.path.join(load_dir, split, "scene.json")
        if os.path.isfile(direct):
            return [os.path.abspath(load_dir)]

        scenes = []
        if not os.path.isdir(load_dir):
            raise FileNotFoundError(f"syn_nuscenes data root does not exist: {load_dir}")
        for name in sorted(os.listdir(load_dir)):
            scene_dir = os.path.join(load_dir, name)
            if os.path.isfile(os.path.join(scene_dir, split, "scene.json")):
                scenes.append(os.path.abspath(scene_dir))
        if not scenes:
            raise FileNotFoundError(
                f"No syn-nuScenes scenes with {split}/scene.json under {load_dir}"
            )
        return scenes

    def convert(self):
        print("Start converting syn_nuscenes ...")
        track_parallel_progress(self.convert_one, self.process_id_list, self.workers)
        print("\nFinished ...")

    def convert_one(self, scene_idx):
        scene_dir = self.scene_dirs[scene_idx]
        with self._open_nusc(scene_dir) as nusc:
            if len(nusc.scene) != 1:
                raise RuntimeError(
                    f"{scene_dir} has {len(nusc.scene)} scenes; expected exactly one"
                )
            scene_data = nusc.get("scene", nusc.scene[0]["token"])
            timestamps = np.asarray(
                self._sensor_timestamps(nusc, scene_data, self.LIDAR_CHANNEL),
                dtype=np.int64,
            )
            self._check_10hz(timestamps, scene_idx)
            scene_name = os.path.basename(scene_dir.rstrip(os.sep))
            meta_path = os.path.join(self._scene_dir(scene_idx), "scene_meta.json")
            with open(meta_path, "w") as fp:
                json.dump(
                    {
                        "scene_name": scene_name,
                        "hz": 10,
                        "num_frames": int(timestamps.shape[0]),
                        "cameras": {str(k): v for k, v in self.CAMERAS.items()},
                        "depth_cam_ids": [
                            cam_id
                            for cam_id in self.CAM_IDS
                            if cam_id not in self.SKIP_DEPTH_CAM_IDS
                        ],
                        "lidar": self.LIDAR_CHANNEL,
                    },
                    fp,
                    indent=2,
                )

            if "images" in self.process_keys:
                self.save_image(nusc, scene_data, scene_idx, timestamps)
                print(f"Processed images for scene {scene_idx:03d} ({scene_name})")
            if "calib" in self.process_keys:
                self.save_calib(nusc, scene_data, scene_idx, timestamps)
                print(f"Processed calib for scene {scene_idx:03d} ({scene_name})")
            if "cam2ego_extrinsics" in self.process_keys:
                self.save_cam2ego_extrinsics(nusc, scene_data, scene_idx, timestamps)
                print(f"Processed cam2ego extrinsics for scene {scene_idx:03d} ({scene_name})")
            if "lidar" in self.process_keys:
                self.save_lidar(nusc, scene_data, scene_idx, timestamps)
                print(f"Processed lidar for scene {scene_idx:03d} ({scene_name})")
            if "ego_pose" in self.process_keys:
                self.save_ego_pose(nusc, scene_data, scene_idx, timestamps)
                print(f"Processed ego poses for scene {scene_idx:03d} ({scene_name})")
            if "depth_map" in self.process_keys:
                self.save_depth_map(nusc, scene_data, scene_idx, timestamps)
                print(f"Processed depth maps for scene {scene_idx:03d} ({scene_name})")

    def __len__(self):
        return len(self.scene_dirs)

    def _scene_dir(self, scene_idx):
        return os.path.join(self.save_dir, f"{scene_idx:03d}")

    @staticmethod
    def _check_10hz(timestamps, scene_idx):
        if timestamps.shape[0] < 2:
            raise RuntimeError(f"scene {scene_idx:03d} has fewer than 2 frames")
        steps = np.diff(timestamps.astype(np.int64))
        if np.any(np.abs(steps - 100000) > 2000):
            raise RuntimeError(
                f"scene {scene_idx:03d} is not 10 Hz "
                f"(timestamp step min={steps.min()} max={steps.max()} us)"
            )

    @contextmanager
    def _open_nusc(self, scene_dir):
        """Open a scene with the nuScenes devkit.

        Annotation and map tables are absent. They are stubbed in a temporary
        dataroot of symlinks so the raw scene is not modified.
        """
        table_root = os.path.join(scene_dir, self.split)
        missing = [
            name
            for name in _STUB_TABLES
            if not os.path.isfile(os.path.join(table_root, name + ".json"))
        ]
        if not missing:
            yield NuScenes(version=self.split, dataroot=scene_dir, verbose=False)
            return

        tmp = tempfile.mkdtemp(prefix="syn_nuscenes_")
        try:
            for name in ("samples", "sweeps"):
                src = os.path.join(scene_dir, name)
                if os.path.isdir(src):
                    os.symlink(os.path.abspath(src), os.path.join(tmp, name))
            version_dir = os.path.join(tmp, self.split)
            os.makedirs(version_dir)
            for name in os.listdir(table_root):
                os.symlink(
                    os.path.abspath(os.path.join(table_root, name)),
                    os.path.join(version_dir, name),
                )
            logs = json.load(open(os.path.join(table_root, "log.json")))
            if "map" in missing:
                os.makedirs(os.path.join(tmp, "maps"), exist_ok=True)
                Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
                    os.path.join(tmp, "maps", "dummy.png")
                )
                json.dump(
                    [
                        {
                            "token": "synmap00000000000000000000000001",
                            "log_tokens": [logs[0]["token"]],
                            "filename": "maps/dummy.png",
                            "category": "semantic_prior",
                        }
                    ],
                    open(os.path.join(version_dir, "map.json"), "w"),
                )
                missing = [name for name in missing if name != "map"]
            for name in missing:
                json.dump([], open(os.path.join(version_dir, name + ".json"), "w"))
            yield NuScenes(version=self.split, dataroot=tmp, verbose=False)
        finally:
            shutil.rmtree(tmp)

    def _sensor_timestamps(self, nusc, scene_data, channel):
        first = nusc.get("sample", scene_data["first_sample_token"])
        if channel not in first["data"]:
            raise KeyError(f"{channel} is not in the first keyframe of {scene_data['name']}")
        current = nusc.get("sample_data", first["data"][channel])
        timestamps = []
        while True:
            timestamps.append(int(current["timestamp"]))
            if current["next"] == "":
                break
            current = nusc.get("sample_data", current["next"])
        return timestamps

    def _closest_tokens(self, nusc, scene_data, timestamps, channel):
        sensor_ts = np.asarray(
            self._sensor_timestamps(nusc, scene_data, channel), dtype=np.int64
        )
        first = nusc.get("sample", scene_data["first_sample_token"])
        current = nusc.get("sample_data", first["data"][channel])
        tokens = []
        while True:
            tokens.append(current["token"])
            if current["next"] == "":
                break
            current = nusc.get("sample_data", current["next"])
        tokens = np.asarray(tokens)
        closest = []
        for timestamp in timestamps:
            idx = int(np.argmin(np.abs(sensor_ts - int(timestamp))))
            if abs(int(sensor_ts[idx]) - int(timestamp)) > 50000:
                raise RuntimeError(
                    f"No {channel} frame within 50 ms of timestamp {int(timestamp)}"
                )
            closest.append(tokens[idx])
        return closest

    @staticmethod
    def _pose_matrix(rotation, translation):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Quaternion(rotation).rotation_matrix
        pose[:3, 3] = np.asarray(translation, dtype=np.float64)
        return pose

    def _cam_to_world(self, nusc, sample_data):
        calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        ego = nusc.get("ego_pose", sample_data["ego_pose_token"])
        cam_to_ego = self._pose_matrix(calib["rotation"], calib["translation"])
        ego_to_world = self._pose_matrix(ego["rotation"], ego["translation"])
        return ego_to_world @ cam_to_ego, calib

    def save_image(self, nusc, scene_data, scene_idx, timestamps):
        for cam_id in self.CAM_IDS:
            tokens = self._closest_tokens(nusc, scene_data, timestamps, self.CAMERAS[cam_id])
            for frame_idx, token in enumerate(tokens):
                cam_data = nusc.get("sample_data", token)
                source = os.path.join(nusc.dataroot, cam_data["filename"])
                target = os.path.join(
                    self._scene_dir(scene_idx),
                    "images",
                    f"{frame_idx:03d}_{cam_id}.jpg",
                )
                shutil.copyfile(source, target)

    def save_calib(self, nusc, scene_data, scene_idx, timestamps):
        for cam_id in self.CAM_IDS:
            tokens = self._closest_tokens(nusc, scene_data, timestamps, self.CAMERAS[cam_id])
            intrinsics_written = False
            for frame_idx, token in enumerate(tokens):
                cam_data = nusc.get("sample_data", token)
                cam_to_world, calib = self._cam_to_world(nusc, cam_data)
                np.savetxt(
                    os.path.join(
                        self._scene_dir(scene_idx),
                        "extrinsics",
                        f"{frame_idx:03d}_{cam_id}.txt",
                    ),
                    cam_to_world,
                )
                if not intrinsics_written:
                    intrinsics = np.asarray(calib["camera_intrinsic"], dtype=np.float64)
                    if intrinsics.size == 0:
                        raise RuntimeError(
                            f"{self.CAMERAS[cam_id]} has no pinhole intrinsic"
                        )
                    ks = [
                        intrinsics[0, 0],
                        intrinsics[1, 1],
                        intrinsics[0, 2],
                        intrinsics[1, 2],
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                    np.savetxt(
                        os.path.join(
                            self._scene_dir(scene_idx),
                            "intrinsics",
                            f"{cam_id}.txt",
                        ),
                        ks,
                    )
                    intrinsics_written = True

    def save_cam2ego_extrinsics(self, nusc, scene_data, scene_idx, timestamps):
        for cam_id in self.CAM_IDS:
            token = self._closest_tokens(
                nusc, scene_data, [int(timestamps[0])], self.CAMERAS[cam_id]
            )[0]
            cam_data = nusc.get("sample_data", token)
            calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            cam_to_ego = self._pose_matrix(calib["rotation"], calib["translation"])
            np.savetxt(
                os.path.join(
                    self._scene_dir(scene_idx),
                    "cam2ego_extrinsics",
                    f"{cam_id}.txt",
                ),
                cam_to_ego,
            )

    def save_ego_pose(self, nusc, scene_data, scene_idx, timestamps):
        tokens = self._closest_tokens(
            nusc, scene_data, timestamps, self.CAMERAS[self.POSE_CAM_ID]
        )
        for frame_idx, token in enumerate(tokens):
            cam_data = nusc.get("sample_data", token)
            ego = nusc.get("ego_pose", cam_data["ego_pose_token"])
            ego_to_world = self._pose_matrix(ego["rotation"], ego["translation"])
            np.savetxt(
                os.path.join(self._scene_dir(scene_idx), "ego_pose", f"{frame_idx:03d}.txt"),
                ego_to_world,
            )

    def save_lidar(self, nusc, scene_data, scene_idx, timestamps):
        tokens = self._closest_tokens(nusc, scene_data, timestamps, self.LIDAR_CHANNEL)
        for frame_idx, token in enumerate(tokens):
            lidar_data = nusc.get("sample_data", token)
            lidar_path = os.path.join(nusc.dataroot, lidar_data["filename"])
            points = LidarPointCloud.from_file(lidar_path).points.T.astype(np.float32)
            points.tofile(
                os.path.join(self._scene_dir(scene_idx), "lidar", f"{frame_idx:03d}.bin")
            )
            calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            ego = nusc.get("ego_pose", lidar_data["ego_pose_token"])
            lidar_to_ego = self._pose_matrix(calib["rotation"], calib["translation"])
            ego_to_world = self._pose_matrix(ego["rotation"], ego["translation"])
            np.savetxt(
                os.path.join(
                    self._scene_dir(scene_idx), "lidar_pose", f"{frame_idx:03d}.txt"
                ),
                ego_to_world @ lidar_to_ego,
            )

    def save_depth_map(self, nusc, scene_data, scene_idx, timestamps):
        lidar_tokens = self._closest_tokens(
            nusc, scene_data, timestamps, self.LIDAR_CHANNEL
        )
        cam_tokens: Dict[int, List[str]] = {
            cam_id: self._closest_tokens(
                nusc, scene_data, timestamps, self.CAMERAS[cam_id]
            )
            for cam_id in self.CAM_IDS
            if cam_id not in self.SKIP_DEPTH_CAM_IDS
        }

        for frame_idx, lidar_token in enumerate(lidar_tokens):
            lidar_data = nusc.get("sample_data", lidar_token)
            lidar_path = os.path.join(nusc.dataroot, lidar_data["filename"])
            lidar_points = LidarPointCloud.from_file(lidar_path).points[:3, :].T
            lidar_calib = nusc.get(
                "calibrated_sensor", lidar_data["calibrated_sensor_token"]
            )
            lidar_ego = nusc.get("ego_pose", lidar_data["ego_pose_token"])
            lidar_to_world = self._pose_matrix(
                lidar_ego["rotation"], lidar_ego["translation"]
            ) @ self._pose_matrix(lidar_calib["rotation"], lidar_calib["translation"])
            lidar_homo = np.concatenate(
                [
                    lidar_points,
                    np.ones((lidar_points.shape[0], 1), dtype=np.float64),
                ],
                axis=1,
            )

            for cam_id, tokens in cam_tokens.items():
                cam_data = nusc.get("sample_data", tokens[frame_idx])
                cam_calib = nusc.get(
                    "calibrated_sensor", cam_data["calibrated_sensor_token"]
                )
                cam_ego = nusc.get("ego_pose", cam_data["ego_pose_token"])
                world_to_ego = np.linalg.inv(
                    self._pose_matrix(cam_ego["rotation"], cam_ego["translation"])
                )
                ego_to_cam = np.linalg.inv(
                    self._pose_matrix(cam_calib["rotation"], cam_calib["translation"])
                )
                cam_points = (ego_to_cam @ world_to_ego @ lidar_to_world @ lidar_homo.T).T
                height = int(cam_data["height"])
                width = int(cam_data["width"])
                depth_map = np.zeros((height, width), dtype=np.float32)

                valid = cam_points[:, 2] > 0
                cam_points = cam_points[valid]
                if cam_points.shape[0] > 0:
                    intrinsics = np.asarray(cam_calib["camera_intrinsic"], dtype=np.float64)
                    pixels = (intrinsics @ cam_points[:, :3].T).T
                    pixels[:, :2] /= np.maximum(pixels[:, 2:3], 1e-6)
                    inside = (
                        (pixels[:, 0] >= 0)
                        & (pixels[:, 0] <= width - 1)
                        & (pixels[:, 1] >= 0)
                        & (pixels[:, 1] <= height - 1)
                    )
                    uv = np.round(pixels[inside, :2]).astype(np.int32)
                    depth = cam_points[inside, 2]
                    if uv.shape[0] > 0:
                        order = np.argsort(depth)
                        u = uv[order, 0]
                        v = uv[order, 1]
                        depth = depth[order]
                        _, unique = np.unique(v * width + u, return_index=True)
                        depth_map[v[unique], u[unique]] = depth[unique].astype(np.float32)

                np.savez_compressed(
                    os.path.join(
                        self._scene_dir(scene_idx),
                        "depth_map",
                        f"{frame_idx:03d}_{cam_id}.npz",
                    ),
                    depth=depth_map,
                    cam_name=self.CAMERAS[cam_id],
                    cam_id=np.int32(cam_id),
                    timestamp=np.int64(timestamps[frame_idx]),
                )

    def create_folder(self):
        for scene_idx in self.process_id_list:
            root = self._scene_dir(scene_idx)
            os.makedirs(root, exist_ok=True)
            if "images" in self.process_keys:
                os.makedirs(os.path.join(root, "images"), exist_ok=True)
                os.makedirs(os.path.join(root, "sky_masks"), exist_ok=True)
            if "calib" in self.process_keys:
                os.makedirs(os.path.join(root, "extrinsics"), exist_ok=True)
                os.makedirs(os.path.join(root, "intrinsics"), exist_ok=True)
            if "cam2ego_extrinsics" in self.process_keys:
                os.makedirs(os.path.join(root, "cam2ego_extrinsics"), exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(os.path.join(root, "lidar"), exist_ok=True)
                os.makedirs(os.path.join(root, "lidar_pose"), exist_ok=True)
            if "ego_pose" in self.process_keys:
                os.makedirs(os.path.join(root, "ego_pose"), exist_ok=True)
            if "depth_map" in self.process_keys:
                os.makedirs(os.path.join(root, "depth_map"), exist_ok=True)
