"""
DDAD数据集完整预处理脚本
参考nuscenes_preprocess.py的保存格式

处理内容：
- 训练集: 150个场景 → data/ddad_process/train (000-149)
- 验证集: 50个场景 → data/ddad_process/valid (000-049)

DDAD坐标系说明：
- extrinsics: sensor-to-vehicle变换 (OpenCV格式)
- pose: sensor-to-world变换
- 深度图: 由DGP自动从LiDAR投影到相机坐标系生成

保存格式：
- intrinsics: 每个相机一个文件 {cam_idx}.txt (静态)
- cam2ego_extrinsics: 每个相机一个文件 {cam_idx}.txt (静态)
- images: {frame}_{cam}.jpg (每帧每相机)
- ego_pose: {frame}.txt (每帧一个)
- depth_map: {frame}_{cam}.npz (每帧每相机)
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from dgp.datasets import SynchronizedSceneDataset


def pose_to_matrix(pose_obj):
    """将DGP Pose对象转换为4x4 numpy矩阵"""
    return pose_obj.matrix.astype(np.float64)


def get_ego_pose_from_lidar(lidar_datum):
    """
    从LiDAR的pose和extrinsics计算ego pose (vehicle-to-world)

    由于DDAD的LiDAR外参是单位矩阵（LiDAR安装在vehicle中心），
    所以LiDAR的pose就是vehicle的pose

    Args:
        lidar_datum: LiDAR数据字典

    Returns:
        4x4 numpy array, ego2world变换矩阵
    """
    lidar_pose = pose_to_matrix(lidar_datum['pose'])
    lidar_ext = pose_to_matrix(lidar_datum['extrinsics'])
    # ego2world = sensor2world @ inv(sensor2ego)
    ego2world = lidar_pose @ np.linalg.inv(lidar_ext)
    return ego2world


def process_one_scene(dataset, scene_idx, scene_dir, cam_list):
    """
    处理单个场景

    Args:
        dataset: SynchronizedSceneDataset对象
        scene_idx: 场景在dataset中的索引
        scene_dir: 保存目录
        cam_list: 相机名称列表
    """
    # 创建目录
    dirs_to_create = [
        'images', 'intrinsics', 'cam2ego_extrinsics',
        'ego_pose', 'depth_map'
    ]
    for d in dirs_to_create:
        os.makedirs(os.path.join(scene_dir, d), exist_ok=True)

    # 获取场景信息
    scene = dataset.scenes[scene_idx]
    num_frames = len(scene.samples)

    # 计算该场景在dataset中的起始索引
    start_idx = sum(len(s.samples) for s in dataset.scenes[:scene_idx])

    # ===== 保存静态数据 (每个相机只有一个) =====
    # 从第一帧获取intrinsics和extrinsics（静态不变）
    first_sample = dataset[start_idx]
    first_datums = first_sample[0]

    for cam_idx, cam_name in enumerate(cam_list):
        cam_datum = first_datums[cam_idx]

        # 保存内参 (每个相机一个文件)
        intrinsics = cam_datum['intrinsics']
        intrinsics_path = os.path.join(scene_dir, 'intrinsics',
                                      f"{str(cam_idx)}.txt")
        # 格式: fx fy cx cy 0 0 0 0 0 (参考nuscenes)
        K = intrinsics
        intrinsics_data = [K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                          0., 0., 0., 0., 0.]
        np.savetxt(intrinsics_path, intrinsics_data)

        # 保存外参 cam2ego (每个相机一个文件，静态不变)
        ext_matrix = pose_to_matrix(cam_datum['extrinsics'])
        ext_path = os.path.join(scene_dir, 'cam2ego_extrinsics',
                               f"{str(cam_idx)}.txt")
        np.savetxt(ext_path, ext_matrix)

    # ===== 保存动态数据 (每帧更新) =====
    for frame_idx in range(num_frames):
        global_idx = start_idx + frame_idx
        sample = dataset[global_idx]
        datums = sample[0]

        # datum顺序: cam_list中的相机 + lidar
        for cam_idx, cam_name in enumerate(cam_list):
            cam_datum = datums[cam_idx]

            # 保存图像 (每帧每相机)
            rgb_image = cam_datum['rgb']
            img_path = os.path.join(scene_dir, 'images',
                                   f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg")
            rgb_image.save(img_path, 'JPEG', quality=95)

            # 保存深度图 (每帧每相机)
            if 'depth' in cam_datum:
                depth = cam_datum['depth']
                depth_path = os.path.join(scene_dir, 'depth_map',
                                         f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.npz")
                np.savez_compressed(depth_path,
                                   depth=depth,
                                   cam_name=cam_name,
                                   cam_id=np.int32(cam_idx),
                                   timestamp=np.int64(cam_datum['timestamp']))

        # 保存ego_pose (每帧一个)
        lidar_datum = datums[len(cam_list)]  # lidar是最后一个
        ego2world = get_ego_pose_from_lidar(lidar_datum)

        ego_pose_path = os.path.join(scene_dir, 'ego_pose',
                                    f"{str(frame_idx).zfill(3)}.txt")
        np.savetxt(ego_pose_path, ego2world)

    return num_frames


def main():
    parser = argparse.ArgumentParser(description='DDAD Dataset Preprocessing')
    parser.add_argument('--data_root', type=str, default='data/raw/ddad_train_val',
                       help='Path to raw DDAD data')
    parser.add_argument('--save_root', type=str, default='data/ddad_process',
                       help='Path to save processed data')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of workers (not used, single process)')
    args = parser.parse_args()

    # DDAD的6个相机
    cam_list = [
        'CAMERA_01',  # 前方
        'CAMERA_05',  # 右前
        'CAMERA_06',  # 右后
        'CAMERA_07',  # 后方
        'CAMERA_08',  # 左后
        'CAMERA_09',  # 左前
    ]

    json_path = os.path.join(args.data_root, 'ddad.json')

    # ===== 处理训练集 =====
    print("=" * 60)
    print("处理训练集 (Train)")
    print("=" * 60)

    train_dataset = SynchronizedSceneDataset(
        json_path,
        datum_names=tuple(cam_list) + ('lidar',),
        generate_depth_from_datum='lidar',
        split='train'
    )

    train_save_dir = os.path.join(args.save_root, 'train')
    os.makedirs(train_save_dir, exist_ok=True)

    print(f"训练集: {len(train_dataset.scenes)} 个场景, {len(train_dataset)} 帧")

    for scene_idx in tqdm(range(len(train_dataset.scenes)),
                         desc="Processing train scenes"):
        scene_dir = os.path.join(train_save_dir, str(scene_idx).zfill(3))
        num_frames = process_one_scene(train_dataset, scene_idx, scene_dir, cam_list)

    print(f"训练集处理完成! 保存到: {train_save_dir}")

    # ===== 处理验证集 =====
    print("\n" + "=" * 60)
    print("处理验证集 (Val)")
    print("=" * 60)

    val_dataset = SynchronizedSceneDataset(
        json_path,
        datum_names=tuple(cam_list) + ('lidar',),
        generate_depth_from_datum='lidar',
        split='val'
    )

    val_save_dir = os.path.join(args.save_root, 'valid')
    os.makedirs(val_save_dir, exist_ok=True)

    print(f"验证集: {len(val_dataset.scenes)} 个场景, {len(val_dataset)} 帧")

    for scene_idx in tqdm(range(len(val_dataset.scenes)),
                         desc="Processing val scenes"):
        scene_dir = os.path.join(val_save_dir, str(scene_idx).zfill(3))
        num_frames = process_one_scene(val_dataset, scene_idx, scene_dir, cam_list)

    print(f"验证集处理完成! 保存到: {val_save_dir}")

    # ===== 统计信息 =====
    print("\n" + "=" * 60)
    print("处理完成统计")
    print("=" * 60)

    # 统计训练集
    train_scenes = len(train_dataset.scenes)
    train_frames = sum(len(s.samples) for s in train_dataset.scenes)

    # 统计验证集
    val_scenes = len(val_dataset.scenes)
    val_frames = sum(len(s.samples) for s in val_dataset.scenes)

    print(f"训练集: {train_scenes} 个场景, {train_frames} 帧")
    print(f"验证集: {val_scenes} 个场景, {val_frames} 帧")
    print(f"总计: {train_scenes + val_scenes} 个场景, {train_frames + val_frames} 帧")
    print()
    print(f"保存位置:")
    print(f"  训练集: {train_save_dir}")
    print(f"  验证集: {val_save_dir}")
    print()
    print("每个场景的文件结构:")
    print("  ├── images/           # {frame}_{cam}.jpg")
    print("  ├── intrinsics/       # {cam}.txt (6个)")
    print("  ├── cam2ego_extrinsics/ # {cam}.txt (6个)")
    print("  ├── ego_pose/         # {frame}.txt")
    print("  └── depth_map/        # {frame}_{cam}.npz")


if __name__ == '__main__':
    main()
