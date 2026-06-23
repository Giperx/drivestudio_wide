# Lyft 3D 目标检测数据集总结

## 概述

Lyft 3D 目标检测数据集由 Lyft 公司通过 Kaggle 竞赛提供。其数据结构与 nuScenes 高度相似，使用相同的 JSON 表格格式和坐标系约定。

**数据位置：** `data/raw/lyft_train/v1.01-train`

工具：`lyft_dataset_sdk`

---

## 数据集统计

| 项目 | 数量 |
|------|------|
| 场景 (Scenes) | 180 |
| 关键帧 (Samples / Keyframes) | 22,680 |
| 数据文件 (Sample Data) | 189,504 |
| 3D 标注框 (Annotations) | 638,179 |
| 可追踪实例 (Instances) | 18,421 |
| 物体类别 (Categories) | 9 |
| 属性 (Attributes) | 18 |

**每个场景：** 固定 126 个关键帧，所有场景一致。

---

## 标注频率

**Lyft 的标注频率是 5 Hz（关键帧间隔 0.2 秒）**

| 数据集 | 标注频率 | 关键帧/场景 | 实际图像帧 |
|--------|----------|-------------|-----------|
| nuScenes | **2 Hz** | ~40帧 | 12Hz (sample_data包含所有帧) |
| Lyft | **5 Hz** | 126帧 | **5Hz (只有关键帧)** |

### 重要区别

**Lyft 只保留了 5Hz 关键帧的图像，没有保留高频原始数据！**

- nuScenes: sample_data 包含所有 12Hz 帧，可通过 `is_key_frame` 字段区分
- Lyft: sample_data 只有 5Hz 关键帧，所有 `is_key_frame` 都是 True
- 189,504 个 sample_data = 7相机×22,680 + LIDAR_TOP×22,680 + LIDAR_FRONT×4,032×2

### 验证结果

```
检查多个场景的标注频率:
Scene 0: 126 frames, 5.02 Hz, interval: 0.100~0.200s
Scene 1: 126 frames, 5.02 Hz, interval: 0.100~0.200s
...

间隔分布:
  0.191~0.201秒 (5Hz): 1240 samples (99.2%)  <-- 主要标注频率
  0.100~0.110秒 (10Hz): 10 samples (0.8%)     <-- 场景边界效应（最后一帧）
```

**结论：** 如果需要 10Hz 数据，Lyft 无法像 nuScenes 那样从原始数据获取，只能通过插值标注。

---

## 目录结构

```
data/raw/lyft_train/
├── train.csv                    # Kaggle 训练 CSV
├── sample_submission.csv        # 提交模板
└── v1.01-train/
    ├── images/                  # 相机图像 (JPEG)
    ├── lidar/                   # 激光雷达点云 (BIN)
    ├── maps/                    # 地图栅格 (map_raster_palo_alto.png)
    └── v1.01-train/             # JSON 标注表
        ├── scene.json
        ├── sample.json
        ├── sample_data.json
        ├── sample_annotation.json
        ├── calibrated_sensor.json
        ├── sensor.json
        ├── ego_pose.json
        ├── category.json
        ├── attribute.json
        ├── instance.json
        ├── visibility.json
        ├── log.json
        └── map.json
```

---

## 传感器

### 相机（7 个）

| 相机 | 说明 |
|------|------|
| CAM_FRONT | 正前方 |
| CAM_FRONT_LEFT | 左前方 |
| CAM_FRONT_RIGHT | 右前方 |
| CAM_BACK_LEFT | 左后方 |
| CAM_BACK_RIGHT | 右后方 |
| CAM_BACK | 正后方 |
| CAM_FRONT_ZOOMED | 前方长焦（变焦镜头） |

**图像分辨率：** 存在两种分辨率组，同一场景内所有相机分辨率一致（CAM_FRONT_ZOOMED 除外）。

| 分辨率 | 场景数 | 标准相机 | CAM_FRONT_ZOOMED |
|--------|--------|----------|------------------|
| 1224x1024 | 148 (82.2%) | 6 个标准相机 | 2048x864 |
| 1920x1080 | 32 (17.8%) | 6 个标准相机 + ZOOMED | 1920x1080 |

**1920x1080 场景（共 32 个）：**
- `host-a101-*`：20 个场景（全部来自车辆 host-a101）
- `host-a102-*`：12 个场景（全部来自车辆 host-a102）

<details>
<summary>点击展开 1920x1080 场景完整列表</summary>

1. host-a101-lidar0-1240710366399037786-1240710391298976894
2. host-a101-lidar0-1240875136198305786-1240875161098795094
3. host-a101-lidar0-1240877587199107226-1240877612099413030
4. host-a101-lidar0-1241216089098610756-1241216113999079830
5. host-a101-lidar0-1241462203298815998-1241462228198805706
6. host-a101-lidar0-1241472407298206026-1241472432198409706
7. host-a101-lidar0-1241561147998866622-1241561172899320654
8. host-a101-lidar0-1241886983298988182-1241887008198992182
9. host-a101-lidar0-1241889710198571346-1241889735098952214
10. host-a101-lidar0-1241893239199111666-1241893264098084346
11. host-a101-lidar0-1242144886399176654-1242144911299066654
12. host-a101-lidar0-1242493624298705334-1242493649198973302
13. host-a101-lidar0-1242580003398722214-1242580028299473214
14. host-a101-lidar0-1242583745399163026-1242583770298821706
15. host-a101-lidar0-1242748817298870302-1242748842198675302
16. host-a101-lidar0-1242748985299274334-1242749010198891466
17. host-a101-lidar0-1242749258298976334-1242749283199254466
18. host-a101-lidar0-1242753236298794334-1242753261198702302
19. host-a101-lidar0-1243095610299140346-1243095635198749774
20. host-a101-lidar0-1243102866399012786-1243102891298922466
21. host-a102-lidar0-1241468916398562586-1241468941298742334
22. host-a102-lidar0-1241548686398885894-1241548711298381586
23. host-a102-lidar0-1241878200398362906-1241878225298546586
24. host-a102-lidar0-1241904536298706586-1241904561198322666
25. host-a102-lidar0-1242150795498255026-1242150820398693830
26. host-a102-lidar0-1242510597398871466-1242510622298829226
27. host-a102-lidar0-1242662270298972894-1242662295198395706
28. host-a102-lidar0-1242684244198410786-1242684269098866094
29. host-a102-lidar0-1242749461398477906-1242749486298996742
30. host-a102-lidar0-1242754954298696742-1242754979198120666
31. host-a102-lidar0-1242755350298764586-1242755375198787666
32. host-a102-lidar0-1242755400298847586-1242755425198579666

</details>

### Train/Val 划分（按分辨率）

已在 `third_party/futr3d/data/lyft/` 下生成 4 个划分文件：

| 文件 | 场景数 | 说明 |
|------|--------|------|
| `train1920.txt` | 27 | 训练集 1920x1080 场景 |
| `train1224.txt` | 123 | 训练集 1224x1024 场景 |
| `val1920.txt` | 5 | 验证集 1920x1080 场景 |
| `val1224.txt` | 25 | 验证集 1224x1024 场景 |

### 预处理数据与车辆对应关系

`data/lyft/processed/` 下的 4 个子目录与车辆的对应关系（通过 cam2ego 外参匹配确认）：

| 目录 | 场景数 | 包含车辆 |
|------|--------|----------|
| `lyft_train1224` | 123 | host-a004(35), host-a011(40), host-a007(23), host-a009(9), host-a008(4), host-a015(4), host-a017(3), host-a006(2), host-a012(2), host-a005(1) |
| `lyft_train1920` | 27 | host-a101(16), host-a102(11) |
| `lyft_val1224` | 25 | host-a011(11), host-a004(7), host-a007(3), host-a015(2), host-a006(1), host-a008(1) |
| `lyft_val1920` | 5 | host-a101(4), host-a102(1) |

**注意：**
- `lyft_train1224` 中的 host-a007 包含 4 个外参不一致的早期场景（Group 3）
- `lyft_train1224` 和 `lyft_val1224` 仅包含 1224x1024 分辨率车辆（10 辆），无 LIDAR_FRONT_LEFT/RIGHT
- `lyft_train1920` 和 `lyft_val1920` 仅包含 1920x1080 分辨率车辆（host-a101/a102），有 3 个激光雷达

---

### 激光雷达（按车辆配置不同）

不同车辆的激光雷达硬件配置不同，与图像分辨率完全对应：

| 激光雷达 | 1224x1024 车辆（10辆） | 1920x1080 车辆（a101/a102） |
|----------|----------------------|---------------------------|
| LIDAR_TOP | 有（每个关键帧） | 有（每个关键帧） |
| LIDAR_FRONT_LEFT | **无** | 有（每个关键帧） |
| LIDAR_FRONT_RIGHT | **无** | 有（每个关键帧） |

**样本数统计：**

| 激光雷达 | 总样本数 | 来源 |
|----------|----------|------|
| LIDAR_TOP | 22,680 | 所有 180 个场景（126 帧/场景） |
| LIDAR_FRONT_LEFT | 4,032 | 仅 host-a101 (20场景) + host-a102 (12场景) = 32 场景 x 126 帧 |
| LIDAR_FRONT_RIGHT | 4,032 | 同上 |

**结论：** LIDAR_FRONT_LEFT/FRONT_RIGHT 样本数较少的原因是**硬件配置差异**——只有 host-a101 和 host-a102 这两辆 1920x1080 分辨率的车安装了 3 个激光雷达，其余 10 辆 1224x1024 的车只安装了 1 个 LIDAR_TOP。

**点云格式：** 每个点 5 个 float（x, y, z, intensity, ring_id）

---

## 坐标系

Lyft 使用与 **nuScenes 相同的坐标约定**：

### 自车坐标系 (Ego Frame)
- **X：** 前方
- **Y：** 左方
- **Z：** 上方

### 相机坐标系（OpenCV 约定）
- **X：** 右方
- **Y：** 下方
- **Z：** 前方（朝向场景内部）

### 激光雷达坐标系
- **X：** 前方
- **Y：** 左方
- **Z：** 上方

### 3D 框约定
- **中心原点：** (0.5, 0.5, 0.5) — 框的几何中心
- **尺寸顺序：** (width, length, height) — 注意：代码中通过 `dims[:, [1, 0, 2]]` 转换为 lwh

### 坐标变换链
```
sensor (cam/lidar) -> ego -> global (world)
```

- **cam2ego：** 来自 `calibrated_sensor` 表（translation + rotation quaternion）
- **ego2global：** 来自 `ego_pose` 表（translation + rotation quaternion）
- **lidar2ego：** 来自 `calibrated_sensor` 表（激光雷达记录）
- **cam-to-world：** `ego2world @ cam2ego`

---

## 相机外参 (cam2ego)

### 每辆车的外参都是唯一的

数据集共 12 辆采集车，**每辆车的相机安装位置都不同**，translation 差异范围 1.1cm ~ 12.3cm。

### 各车辆相机 FOV 统计

#### 1224x1024 分辨率车辆 (host-a004 ~ a017)

| 车辆 | 相机 | 焦距 (fx=fy) | 水平FOV | 垂直FOV |
|------|------|--------------|---------|---------|
| **host-a004** | CAM_FRONT | 881.44 | 69.55° | 60.30° |
| | CAM_FRONT_LEFT | 879.00 | 69.70° | 60.44° |
| | CAM_FRONT_RIGHT | 881.87 | 69.52° | 60.28° |
| | CAM_BACK_LEFT | 880.83 | 69.58° | 60.34° |
| | CAM_BACK_RIGHT | 879.12 | 69.69° | 60.43° |
| | CAM_BACK | 884.09 | 69.39° | 60.15° |
| | CAM_FRONT_ZOOMED | 3418.76 | 33.35° | 14.40° |
| **host-a005** | CAM_FRONT | 882.62 | 69.47° | 60.24° |
| | CAM_FRONT_LEFT | 881.90 | 69.52° | 60.28° |
| | CAM_FRONT_RIGHT | 883.51 | 69.42° | 60.18° |
| | CAM_BACK_LEFT | 882.43 | 69.49° | 60.25° |
| | CAM_BACK_RIGHT | 881.28 | 69.56° | 60.31° |
| | CAM_BACK | 883.76 | 69.40° | 60.17° |
| | CAM_FRONT_ZOOMED | 3433.02 | 33.22° | 14.34° |
| **host-a006** | CAM_FRONT | 877.37 | 69.79° | 60.53° |
| | CAM_FRONT_LEFT | 877.41 | 69.79° | 60.53° |
| | CAM_FRONT_RIGHT | 876.85 | 69.83° | 60.56° |
| | CAM_BACK_LEFT | 876.43 | 69.85° | 60.59° |
| | CAM_BACK_RIGHT | 877.07 | 69.81° | 60.55° |
| | CAM_BACK | 877.18 | 69.81° | 60.54° |
| | CAM_FRONT_ZOOMED | 3416.80 | 33.37° | 14.41° |
| **host-a007** | CAM_FRONT | 881.16 | 69.56° | 60.32° |
| | CAM_FRONT_LEFT | 880.16 | 69.62° | 60.37° |
| | CAM_FRONT_RIGHT | 878.92 | 69.70° | 60.44° |
| | CAM_BACK_LEFT | 881.38 | 69.55° | 60.31° |
| | CAM_BACK_RIGHT | 882.65 | 69.47° | 60.23° |
| | CAM_BACK | 881.41 | 69.55° | 60.30° |
| | CAM_FRONT_ZOOMED | 3409.18 | 33.44° | 14.44° |
| **host-a008** | CAM_FRONT | 879.51 | 69.66° | 60.41° |
| | CAM_FRONT_LEFT | 876.90 | 69.82° | 60.56° |
| | CAM_FRONT_RIGHT | 877.66 | 69.78° | 60.52° |
| | CAM_BACK_LEFT | 877.73 | 69.77° | 60.51° |
| | CAM_BACK_RIGHT | 879.76 | 69.65° | 60.40° |
| | CAM_BACK | 878.76 | 69.71° | 60.45° |
| | CAM_FRONT_ZOOMED | 3429.45 | 33.25° | 14.36° |
| **host-a009** | CAM_FRONT | 879.36 | 69.67° | 60.42° |
| | CAM_FRONT_LEFT | 874.88 | 69.95° | 60.67° |
| | CAM_FRONT_RIGHT | 875.81 | 69.89° | 60.62° |
| | CAM_BACK_LEFT | 879.70 | 69.65° | 60.40° |
| | CAM_BACK_RIGHT | 877.71 | 69.77° | 60.51° |
| | CAM_BACK | 879.90 | 69.64° | 60.39° |
| | CAM_FRONT_ZOOMED | 3411.60 | 33.41° | 14.43° |
| **host-a011** | CAM_FRONT | 884.24 | 69.38° | 60.14° |
| | CAM_FRONT_LEFT | 882.48 | 69.48° | 60.24° |
| | CAM_FRONT_RIGHT | 880.69 | 69.59° | 60.34° |
| | CAM_BACK_LEFT | 882.54 | 69.48° | 60.24° |
| | CAM_BACK_RIGHT | 882.95 | 69.45° | 60.22° |
| | CAM_BACK | 882.94 | 69.45° | 60.22° |
| | CAM_FRONT_ZOOMED | 3421.14 | 33.33° | 14.39° |
| **host-a012** | CAM_FRONT | 883.62 | 69.41° | 60.18° |
| | CAM_FRONT_LEFT | 874.51 | 69.97° | 60.70° |
| | CAM_FRONT_RIGHT | 880.17 | 69.62° | 60.37° |
| | CAM_BACK_LEFT | 880.38 | 69.61° | 60.36° |
| | CAM_BACK_RIGHT | 881.84 | 69.52° | 60.28° |
| | CAM_BACK | 883.29 | 69.43° | 60.20° |
| | CAM_FRONT_ZOOMED | 3440.21 | 33.15° | 14.31° |
| **host-a015** | CAM_FRONT | 875.89 | 69.89° | 60.62° |
| | CAM_FRONT_LEFT | 876.00 | 69.88° | 60.61° |
| | CAM_FRONT_RIGHT | 878.06 | 69.75° | 60.49° |
| | CAM_BACK_LEFT | 876.41 | 69.85° | 60.59° |
| | CAM_BACK_RIGHT | 876.87 | 69.83° | 60.56° |
| | CAM_BACK | 877.26 | 69.80° | 60.54° |
| | CAM_FRONT_ZOOMED | 3415.28 | 33.38° | 14.42° |
| **host-a017** | CAM_FRONT | 876.78 | 69.83° | 60.57° |
| | CAM_FRONT_LEFT | 876.36 | 69.86° | 60.59° |
| | CAM_FRONT_RIGHT | 874.85 | 69.95° | 60.68° |
| | CAM_BACK_LEFT | 876.52 | 69.85° | 60.58° |
| | CAM_BACK_RIGHT | 878.02 | 69.75° | 60.50° |
| | CAM_BACK | 881.47 | 69.54° | 60.30° |
| | CAM_FRONT_ZOOMED | 3385.56 | 33.66° | 14.54° |

#### 1920x1080 分辨率车辆 (host-a101, host-a102)

| 车辆 | 相机 | 焦距 (fx=fy) | 水平FOV | 垂直FOV |
|------|------|--------------|---------|---------|
| **host-a101** | CAM_FRONT | 1109.05 | 81.76° | 51.92° |
| | CAM_FRONT_LEFT | 1110.77 | 81.67° | 51.85° |
| | CAM_FRONT_RIGHT | 1108.78 | 81.77° | 51.93° |
| | CAM_BACK_LEFT | 1110.20 | 81.70° | 51.88° |
| | CAM_BACK_RIGHT | 1110.48 | 81.69° | 51.87° |
| | CAM_BACK | 1112.84 | 81.57° | 51.77° |
| | CAM_FRONT_ZOOMED | 3962.24 | 27.24° | 15.52° |
| **host-a102** | CAM_FRONT | 1109.36 | 81.74° | 51.91° |
| | CAM_FRONT_LEFT | 1107.66 | 81.83° | 51.98° |
| | CAM_FRONT_RIGHT | 1107.10 | 81.86° | 52.00° |
| | CAM_BACK_LEFT | 1105.76 | 81.93° | 52.06° |
| | CAM_BACK_RIGHT | 1109.01 | 81.76° | 51.92° |
| | CAM_BACK | 1110.77 | 81.67° | 51.85° |
| | CAM_FRONT_ZOOMED | 3986.36 | 27.08° | 15.43° |

### FOV 汇总

| 分辨率组 | 标准相机 FOV | CAM_FRONT_ZOOMED FOV |
|----------|-------------|---------------------|
| 1224x1024 | 水平 ~69.5°-70.0°, 垂直 ~60.1°-60.7° | 水平 ~33.2°-33.7°, 垂直 ~14.3°-14.5° |
| 1920x1080 | 水平 ~81.6°-81.9°, 垂直 ~51.8°-52.1° | 水平 ~27.1°-27.2°, 垂直 ~15.4°-15.5° |

| 车辆 | 场景数 | 分辨率 | 外参一致性 | 备注 |
|------|--------|--------|-----------|------|
| host-a004 | 42 | 1224x1024 | 一致 | — |
| host-a005 | 1 | 1224x1024 | 一致 | 仅 1 个场景 |
| host-a006 | 3 | 1224x1024 | 一致 | — |
| **host-a007** | 26 | 1224x1024 | **不一致** | 4 个早期场景外参不同（见下文） |
| host-a008 | 5 | 1224x1024 | 一致 | — |
| **host-a009** | 9 | 1224x1024 | **不一致** | 3 种不同外参 |
| host-a011 | 51 | 1224x1024 | 一致 | 场景数最多 |
| host-a012 | 2 | 1224x1024 | 一致 | — |
| host-a015 | 6 | 1224x1024 | 一致 | — |
| host-a017 | 3 | 1224x1024 | 一致 | — |
| **host-a101** | 20 | **1920x1080** | 一致 | — |
| **host-a102** | 12 | **1920x1080** | 一致 | — |

### host-a007 外参不一致详情

host-a007 有 3 组不同的 CAM_FRONT 外参，其中 Group 1 和 Group 2 数值相同（仅 calibrated_sensor token 不同），Group 3 真正不同：

| 分组 | 场景数 | translation | 说明 |
|------|--------|-------------|------|
| Group 1 | 4 | [1.504, -0.039, **1.718**] | 主要外参 |
| Group 2 | 18 | [1.504, -0.039, **1.718**] | 同 Group 1 |
| Group 3 | 4 | [1.498, -0.036, **1.662**] | z 轴差 ~5.6cm |

Group 3 的 4 个场景（时间戳 1230x 开头，为早期采集）：
- host-a007-lidar0-1230936221299185986-1230936246198612066
- host-a007-lidar0-1230672860198383106-1230672885099108186
- host-a007-lidar0-1230485630199365106-1230485655099030186
- host-a007-lidar0-1230485630199365106-1230485655099030186

### host-a009 外参不一致详情

| 分组 | 场景数 | translation | rotation |
|------|--------|-------------|----------|
| Group 1 | 4 | [1.537, -0.043, 1.651] | [0.506, -0.496, 0.495, -0.502] |
| Group 2 | 4 | [1.547, -0.054, 1.650] | [0.507, -0.497, 0.495, -0.501] |
| Group 3 | 1 | [1.518, -0.029, 1.694] | [0.506, -0.496, 0.498, -0.500] |

### 车辆间外参差异（CAM_FRONT translation，单位 cm）

| 车辆对比 | 差异 |
|----------|------|
| host-a004 vs host-a005 | 1.1 cm |
| host-a004 vs host-a102 | 1.3 cm |
| host-a004 vs host-a011 | 2.0 cm |
| host-a101 vs host-a102 | 2.0 cm |
| host-a006 vs host-a007 | 12.3 cm |
| host-a006 vs host-a009 | 9.9 cm |
| host-a006 vs host-a011 | 9.4 cm |

**结论：** 车辆间外参差异主要来自相机物理安装位置不同，差异在 1~12cm 量级。

---

## 标注

### 物体类别（9 类）

| 类别 | 数量 | 说明 |
|------|------|------|
| car | 534,911 | 小汽车 |
| other_vehicle | 33,376 | 其他车辆 |
| pedestrian | 24,935 | 行人 |
| bicycle | 20,928 | 自行车（含/不含骑手） |
| truck | 14,164 | 卡车 |
| bus | 8,729 | 公交车 |
| motorcycle | 818 | 摩托车 |
| animal | 186 | 动物 |
| emergency_vehicle | 132 | 紧急车辆 |

### 标注字段

每个标注包含：
- `token`：唯一 ID
- `sample_token`：关联的关键帧
- `instance_token`：追踪 ID（跨帧关联同一物体）
- `translation`：[x, y, z] 全局坐标系下的中心位置
- `size`：[width, length, height]
- `rotation`：四元数 [x, y, z, w]
- `num_lidar_pts`：框内激光雷达点数
- `num_radar_pts`：框内毫米波雷达点数
- `attribute_tokens`：属性 ID 列表
- `visibility_token`：可见性等级
- `prev`/`next`：同一实例的前后帧链接

### 动态属性（与 nuScenes 类似）

Lyft 有 **18 种属性**，描述物体的运动状态。与 nuScenes 不同的是，Lyft **没有 velocity 字段**，而是通过属性标注来描述行为。

| 属性 | 数量 | 说明 |
|------|------|------|
| is_stationary | 321,981 | 静止不动 |
| object_action_parked | 257,939 | 停放状态 |
| object_action_driving_straight_forward | 244,805 | 直行 |
| object_action_stopped | 94,970 | 临时停车（如等红灯） |
| object_action_walking | 17,890 | 行走中 |
| object_action_right_turn | 6,694 | 右转 |
| object_action_standing | 5,332 | 站立 |
| object_action_left_turn | 5,074 | 左转 |
| object_action_lane_change_left | 1,463 | 向左变道 |
| object_action_lane_change_right | 1,370 | 向右变道 |
| object_action_running | 621 | 跑步 |
| object_action_sitting | 586 | 坐着 |
| object_action_other_motion | 582 | 其他运动 |
| object_action_u_turn | 407 | 掉头 |
| object_action_reversing | 278 | 倒车 |
| object_action_gliding_on_wheels | 165 | 滑行（滑板、轮滑等） |
| object_action_abnormal_or_traffic_violation | 2 | 异常行为/违章 |
| object_action_loss_of_control | 1 | 失控 |

**动态判断方式：**
- 有 `is_stationary` 或 `object_action_parked` → 静态
- 有其他 `object_action_*`（如 driving, walking, turning）→ 动态

### 实例追踪

- **18,421 个实例**（唯一可追踪物体）
- **17,996 个实例**出现在 2 帧以上（可追踪）
- **每个实例最大标注数：** 126（等于单场景关键帧数）

### 可见性等级

| Token | 等级 | 说明 |
|-------|------|------|
| v0-40 | 1 | 可见度 0-40% |
| v40-60 | 2 | 可见度 40-60% |
| v60-80 | 3 | 可见度 60-80% |
| v80-100 | 4 | 可见度 80-100% |

---

## 与 nuScenes 对比

| 特性 | nuScenes | Lyft |
|------|----------|------|
| JSON 表格 | 是 | 是（格式相同） |
| **标注频率** | **2 Hz** | **5 Hz** |
| **sample_data** | **12Hz 所有帧** | **只有 5Hz 关键帧** |
| 相机数量 | 6 | 7（多了 CAM_FRONT_ZOOMED） |
| 激光雷达数量 | 1 (LIDAR_TOP) | 1 或 3（取决于车辆，见上文） |
| 物体类别 | 23 | 9 |
| 动态属性 | 有（visibility, activity） | 有（18 种 action 属性） |
| 标注中的速度 | 有（velocity 字段） | **无** |
| 实例追踪 | 有 | 有 |
| 坐标系 | OpenCV 相机，FLU 自车 | 与 nuScenes 相同 |
| 框中心原点 | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) |
| 框尺寸顺序 | w, l, h | w, l, h |
| 点云格式 | 5D (x,y,z,intensity,ring) | 5D (x,y,z,intensity,ring) |
| **能否获取高频数据** | ✓ 能（12Hz sample_data） | ✗ 不能（只有5Hz关键帧） |

**主要区别：**
1. **标注频率不同**：Lyft 是 5Hz，nuScenes 是 2Hz
2. **sample_data 内容不同**：nuScenes 包含所有 12Hz 帧，Lyft 只有 5Hz 关键帧
3. Lyft 有 7 个相机（多了 CAM_FRONT_ZOOMED），nuScenes 6 个
4. Lyft 激光雷达数量因车辆而异：1920x1080 车辆有 3 个（TOP + FRONT_LEFT + FRONT_RIGHT），1224x1024 车辆仅 1 个（TOP）；nuScenes 统一 1 个
5. Lyft 类别更少（9 vs 23），但动作属性更丰富
6. Lyft 标注中**没有 velocity 字段**（nuScenes 有）
7. 所有数据来自美国加州 Palo Alto

### nuScenes 获取高频数据的方法

nuScenes 通过遍历 sample_data 表获取 12Hz 图像帧：
```python
# nuScenes: sample_data 包含所有 12Hz 帧
cur_img_data = nusc.get('sample_data', first_sample_record['data'][cam_name])
while True:
    # 获取所有帧（包括非关键帧）
    cams_meta[cam_name]["timestamps"].append(cur_img_data['timestamp'])
    cams_meta[cam_name]["is_key_frame"].append(cur_img_data['is_key_frame'])
    cur_img_data = nusc.get('sample_data', cur_img_data['next'])
```

Lyft 无法使用此方法，因为 sample_data 只有关键帧。

---

## 在 DriveStudio 中的使用

### 现有实现

`third_party/futr3d/mmdet3d/datasets/lyft_dataset.py` 中的 `LyftDataset` 类：
- 加载预处理的 `.pkl` 标注文件
- 支持相机和激光雷达模态
- 将框中心从 (0.5, 0.5, 0.5) 转换为 KITTI 风格 (0.5, 0.5, 0)
- 使用 `LiDARInstance3DBoxes` 表示 3D 框
- 通过 `dims[:, [1, 0, 2]]` 将框尺寸从 wlh 转换为 lwh

### 数据准备

要在 DriveStudio 中使用 Lyft，需要：
1. 运行 futr3d 数据转换器生成 `.pkl` info 文件
2. 或者适配 nuScenes 预处理流程（因为 JSON 格式相同）

### 激光雷达修复

`lyft_fix_lidar.py` 脚本修复了一个已知问题：部分激光雷达文件只有 4 个 float 而非 5 个。当前数据集 (v1.01) 已经应用了修复。

---

## 自车轨迹 (Ego Pose)

- **总 ego pose 数：** 177,789
- **格式：** Translation [x, y, z] + Rotation 四元数 [x, y, z, w]
- **采集地点：** 全部来自美国加州 Palo Alto

---

## 地图

- **map_raster_palo_alto.png** — Palo Alto 驾驶区域的栅格地图

---

## DriveStudio 集成注意事项

1. **JSON schema 与 nuScenes 相同** — 预处理代码大部分可复用
2. **7 个相机** — 需要处理额外的 CAM_FRONT_ZOOMED
3. **激光雷达数量因车辆而异** — 1224x1024 车辆仅 LIDAR_TOP，1920x1080 车辆有 3 个；建议统一使用 LIDAR_TOP 作为主激光雷达（与 nuScenes 一致）
4. **无 velocity** — 无法通过速度计算动态遮罩，需使用 attribute_tokens 判断动态/静态
5. **丰富的动作属性** — 可通过 `is_stationary` 和 `object_action_*` 属性推导动态/静态状态
6. **框格式** — wlh 顺序，中心在 (0.5, 0.5, 0.5)
7. **外参按车辆区分** — 12 辆车各自外参不同，同一车辆内基本一致（host-a007 和 host-a009 有少量不一致场景）
8. **两种分辨率** — 1920x1080 (host-a101/a102) 和 1224x1024 (其余车辆)，预处理时需分别处理

---

## 附录：预处理数据场景号与车辆对应关系

`data/lyft/processed/` 下各子目录的场景号（000, 001, ...）与车辆的完整对应关系：

### lyft_train1224（123 场景）

| 车辆 | 场景数 | 场景号 |
|------|--------|--------|
| host-a004 | 35 | 009  012  018  019  025  027  030  039  042  045  046  048  049  050  051  054  055  056  065  070  074  075  077  080  085  089  090  091  094  096  111  112  113  116  122 |
| host-a011 | 40 | 001  003  007  008  010  013  014  016  017  024  029  034  036  037  038  041  043  044  047  053  060  061  062  064  066  069  078  079  087  088  093  099  106  107  108  114  115  117  118  119 |
| host-a007 | 23 | 011  015  020  023  026  032  033  040  052  057  059  067  072  073  076  083  084  086  098  102  103  110  121 |
| host-a009 | 9 | 004  006  022  035  068  097  104  105  120 |
| host-a008 | 4 | 005  028  058  063 |
| host-a015 | 4 | 021  082  095  100 |
| host-a017 | 3 | 071  081  092 |
| host-a006 | 2 | 000  031 |
| host-a012 | 2 | 002  101 |
| host-a005 | 1 | 109 |

### lyft_train1920（27 场景）

| 车辆 | 场景数 | 场景号 |
|------|--------|--------|
| host-a101 | 16 | 000  001  006  008  009  010  011  012  013  016  017  018  019  020  021  026 |
| host-a102 | 11 | 002  003  004  005  007  014  015  022  023  024  025 |

### lyft_val1224（25 场景）

| 车辆 | 场景数 | 场景号 |
|------|--------|--------|
| host-a011 | 11 | 002  005  008  009  013  014  015  016  017  018  023 |
| host-a004 | 7 | 000  001  004  006  007  012  024 |
| host-a007 | 3 | 010  019  020 |
| host-a015 | 2 | 011  021 |
| host-a006 | 1 | 003 |
| host-a008 | 1 | 022 |

### lyft_val1920（5 场景）

| 车辆 | 场景数 | 场景号 |
|------|--------|--------|
| host-a101 | 4 | 000  001  003  004 |
| host-a102 | 1 | 002 |
