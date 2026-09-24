# syn-nuScenes

CARLA 生成的虚拟 nuScenes 场景。目录和表结构与官方 nuScenes 相同，可以用 `nuscenes-devkit` 读取，再导出成 DriveStudio 现有的 10 Hz 训练格式。

当前样例是 `data/syn_nuscenes/Town10HD_scene_0001`：Town10HD，20 秒，原生 10 Hz，200 帧。场景按静态处理，没有 3D box，也不做 dynamic mask。

## 1. 原始数据

每个场景自己就是一个 nuScenes dataroot，不要把多个场景摊平进同一个 `samples/`。

```text
data/syn_nuscenes/
  └── Town10HD_scene_0001/
       ├── samples/          # 2 Hz keyframe，每个通道 40 帧
       ├── sweeps/           # 其余 8 Hz，每个通道 160 帧
       ├── v1.0-trainval/    # scene / sample / sample_data / ego_pose / sensor / calibrated_sensor / log
       ├── metadata/         # CARLA 采集配置，预处理不读
       └── masks/C_CYL/      # 柱面有效区域，当前不用
```

`sample.json` 只有 40 个 2 Hz keyframe。图像、LiDAR、ego pose 的完整 10 Hz 序列在 `sample_data` 的 `next` 链上，keyframe 在 `samples/`，中间帧在 `sweeps/`。时间步长约 100000 us。

标注表是空的：没有 `sample_annotation`、`instance`、`category`。官方 devkit 启动时仍会要这些表和一张 map。预处理会在临时目录里补空表，不改原始数据。

采集配置里写了 5 辆车和 3 个行人，但没有框。当前流水线不使用它们。

## 2. 相机

原始有 5 个相机和 1 个 LiDAR。柱面相机 `CAM_BACK_WIDE_CYL_GT` 没有针孔内参，不导出。

导出编号沿用 nuScenes 后向相机的槽位，不是从 0 重新编号：

| id | channel | 分辨率 | 内参 | depth |
|----|---------|--------|------|-------|
| 5 | `CAM_BACK_SOURCE` | 1920×1080 | fx=fy=1662.77, cx=960, cy=540 | 生成 |
| 4 | `CAM_BACK_RIGHT_SOURCE` | 1920×1080 | 同上 | 生成 |
| 3 | `CAM_BACK_LEFT_SOURCE` | 1920×1080 | 同上 | 生成 |
| 2 | `CAM_BACK_WIDE_RECT_GT` | 5760×1080 | fx=fy=1662.77, cx=2880, cy=540 | 不生成 |

三个 source 相机水平 FOV 60°，相邻重叠约 30°。`CAM_BACK_WIDE_RECT_GT` 是 120° 的针孔宽图，和 `CAM_BACK_SOURCE` 共安装位。id 2 只保留图像、外参、内参和 cam-to-ego，不投影 LiDAR depth。

`LIDAR_TOP` 是 32 线，约 50 万点/帧，文件为 nuScenes 的 `(x, y, z, intensity, ring)` float32。导出时只保留前 4 维，和现有 loader 的 `reshape(-1, 4)` 一致。

## 3. 和官方 nuScenes 预处理的差别

| | 官方 nuScenes | syn-nuScenes |
|--|---------------|--------------|
| 标注关键帧 | 2 Hz | 2 Hz keyframe，但 sweeps 已是真实 10 Hz |
| 10 Hz | `interpolate_N=4` 插值 | 直接走 `sample_data` 链，不插值 |
| 动态物体 | mask + 3D box | 不生成。传入这两个 key 会被跳过 |
| 相机 | 6 个环视，id 0–5 | 上表 4 个，没有 id 0 和 1 |
| 场景编号 | nuScenes scene 列表下标，例如 037 | `data/syn_nuscenes/` 下场景文件夹的排序下标 |

`start_idx` 不是官方 scene id。当前只有 `Town10HD_scene_0001`，它的下标是 `0`。

## 4. 预处理

依赖在 `drivestudio` 环境里（`nuscenes-devkit`、`pyquaternion`）。`Gendata` 环境没有这些包，不要用它跑。

```shell
export PYTHONPATH=/home/test/LIVA/XZP/FeedForward/drivestudio
conda activate drivestudio

python datasets/preprocess.py \
    --data_root data/syn_nuscenes \
    --target_dir data/syn_nuscenes/processed \
    --dataset syn_nuscenes \
    --split v1.0-trainval \
    --start_idx 0 \
    --num_scenes 1 \
    --interpolate_N 0 \
    --workers 1 \
    --process_keys cam2ego_extrinsics ego_pose depth_map images lidar calib
```

`target_dir` 里的 `processed` 会被换成 `processed_10Hz`。上面这条命令的结果在：

```text
data/syn_nuscenes/processed_10Hz/trainval/000/
```

`000` 是场景下标。`scene_meta.json` 里记录原始文件夹名 `Town10HD_scene_0001`。

多个场景按文件夹名字典序编号。例如再放入 `Town10HD_scene_0002` 后，它是下标 `1`。

## 5. 导出结构

```text
data/syn_nuscenes/processed_10Hz/trainval/000/
  ├── images/                 # {frame:03d}_{cam_id}.jpg，200 × 4
  ├── extrinsics/             # cam-to-world，{frame:03d}_{cam_id}.txt
  ├── intrinsics/             # fx fy cx cy k1 k2 p1 p2 k3，{cam_id}.txt
  ├── cam2ego_extrinsics/     # {cam_id}.txt
  ├── ego_pose/               # ego-to-world，{frame:03d}.txt，取相机 5 的时刻
  ├── lidar/                  # {frame:03d}.bin，float32 (N, 4)，点在 LiDAR 坐标系
  ├── lidar_pose/             # lidar-to-world，{frame:03d}.txt
  ├── depth_map/              # {frame:03d}_{cam_id}.npz，只有相机 5、4、3
  └── scene_meta.json
```

没有 `dynamic_masks/`、`instances/`、`humanpose/`。depth 是把当前帧 LiDAR 投到针孔图像上的稀疏图，像素冲突时保留更近的深度。

图像文件沿用项目的 `.jpg` 文件名，内容仍是原始 PNG 字节。

## 6. 训练配置

`configs/datasets/syn_nuscenes/4cams.yaml`

- `dataset: syn_nuscenes`
- `data_root: data/syn_nuscenes/processed_10Hz/trainval`
- `cameras: [5, 4, 3, 2]`
- `load_dynamic_mask: False`
- `load_sky_mask: False`
- `load_smpl: False`

加载器仍是 `NuScenesPixelSource` / `NuScenesLiDARSource`。官方数据用相机 0 做世界对齐；这里没有相机 0，会改用该帧已导出外参里编号最小的相机。没有 `instances_info.json` 时会建成空 instance，后续动态节点为空。

## 7. Wide GT

相机 2（`CAM_BACK_WIDE_RECT_GT`，5760×1080）是全像素 GT，不生成 mask。`tools/export_syn_nuscenes_wide_gt.py` 把它 resize 成和 nuScenes / Lyft wide 图相同的两档尺寸，文件名沿用 `{帧}_{5}_wide.png`。`5` 是这套 wide 图的固定槽位，不是源相机编号。

```shell
python tools/export_syn_nuscenes_wide_gt.py \
    --processed_dir data/syn_nuscenes/processed_10Hz/trainval \
    --output_root data/syn_nuscenes
```

默认尺寸是 `1554x294` 和 `1344x252`（宽×高）。`1344x252` 与原图 16:3 一致；`1554x294` 与现有 sparse wide 图的像素尺寸一致，直接缩放，不裁剪。

```text
data/syn_nuscenes/
  ├── syn_nuscenes_wide_gt_1554x294/
  │    └── 000/rgb/000_5_wide.png
  └── syn_nuscenes_wide_gt_1344x252/
       └── 000/rgb/000_5_wide.png
```

这两个目录在 `data/syn_nuscenes/` 下，已被 gitignore 覆盖。新场景跑完预处理后，同一条命令会按场景编号再导出一份。
