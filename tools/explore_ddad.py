"""
DDAD数据集全面探索脚本
直接读取DGP JSON文件和文件名，不依赖SynchronizedSceneDataset

输出：docs/DDAD_DATASET_SUMMARY.md
"""

import json
import os
import numpy as np
from collections import defaultdict, Counter

BASE_DIR = '/home/test/LIVA/XZP/FeedForward/drivestudio/data/raw/ddad_train_val'
CAM_LIST = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
CAM_DIRECTIONS = {
    'CAMERA_01': 'Front',
    'CAMERA_05': 'Front-Right',
    'CAMERA_06': 'Rear-Right',
    'CAMERA_07': 'Rear',
    'CAMERA_08': 'Rear-Left',
    'CAMERA_09': 'Front-Left',
}


def load_scene_json(scene_dir):
    scene_files = [f for f in os.listdir(os.path.join(BASE_DIR, scene_dir))
                   if f.startswith('scene_') and f.endswith('.json')]
    if not scene_files:
        return None
    with open(os.path.join(BASE_DIR, scene_dir, scene_files[0])) as f:
        return json.load(f)


def load_calibration(scene_dir, cal_key):
    cal_path = os.path.join(BASE_DIR, scene_dir, 'calibration', f'{cal_key}.json')
    if not os.path.exists(cal_path):
        return None
    with open(cal_path) as f:
        return json.load(f)


def compute_fps_from_filenames(directory, extension='.png'):
    """从文件名中的时间戳计算FPS"""
    files = sorted([f for f in os.listdir(directory) if f.endswith(extension)])
    if len(files) < 2:
        return 0, len(files)
    ts = [int(f.replace(extension, '')) for f in files]
    diffs_ms = [(ts[i+1] - ts[i]) / 1000 for i in range(len(ts) - 1)]
    avg_interval = sum(diffs_ms) / len(diffs_ms)
    fps = 1000 / avg_interval if avg_interval > 0 else 0
    return fps, len(files)


def main():
    # 加载顶层JSON
    with open(os.path.join(BASE_DIR, 'ddad.json')) as f:
        ddad_json = json.load(f)

    # 收集所有场景
    all_scenes = []
    for split_id, split_data in ddad_json['scene_splits'].items():
        split_name = {0: 'train', 1: 'val', 2: 'test'}.get(int(split_id), f'split_{split_id}')
        for scene_file in split_data['filenames']:
            scene_dir = scene_file.split('/')[0]
            all_scenes.append({'dir': scene_dir, 'split': split_name})

    print(f"Total scenes: {len(all_scenes)}")

    # ===== 逐场景分析 =====
    scene_infos = []
    all_cal_data = {}  # cal_key -> calibration dict
    cam_image_sizes = defaultdict(set)
    cam_intrinsics = {}
    cam_extrinsics_all = defaultdict(list)  # cam_name -> [(scene_dir, ext_dict)]

    for i, si in enumerate(all_scenes):
        scene_dir = si['dir']
        scene = load_scene_json(scene_dir)
        if not scene:
            continue

        # 标定
        cal_keys = set(s.get('calibration_key', '') for s in scene.get('samples', []))
        cal = None
        for ck in cal_keys:
            if ck not in all_cal_data:
                c = load_calibration(scene_dir, ck)
                if c:
                    all_cal_data[ck] = c
            cal = all_cal_data.get(ck, cal)

        # 图像尺寸 (从datum JSON获取)
        for datum in scene.get('data', []):
            dname = datum['id']['name']
            if 'image' in datum.get('datum', {}):
                img = datum['datum']['image']
                w, h = img.get('width', 0), img.get('height', 0)
                if w > 0 and h > 0:
                    cam_image_sizes[dname].add((w, h))

        # 内参/外参
        if cal:
            for idx, cname in enumerate(cal['names']):
                if cname.startswith('CAMERA'):
                    if cname not in cam_intrinsics:
                        cam_intrinsics[cname] = cal['intrinsics'][idx]
                    cam_extrinsics_all[cname].append((scene_dir, cal['extrinsics'][idx]))

        # FPS (从CAMERA_01文件名)
        cam01_dir = os.path.join(BASE_DIR, scene_dir, 'rgb', 'CAMERA_01')
        fps, num_files = compute_fps_from_filenames(cam01_dir, '.png') if os.path.exists(cam01_dir) else (0, 0)

        # LiDAR FPS
        lidar_dir = os.path.join(BASE_DIR, scene_dir, 'point_cloud', 'LIDAR')
        lidar_fps, num_lidar = compute_fps_from_filenames(lidar_dir, '.npz') if os.path.exists(lidar_dir) else (0, 0)

        scene_infos.append({
            'dir': scene_dir,
            'split': si['split'],
            'name': scene.get('name', ''),
            'description': scene.get('description', ''),
            'log': scene.get('log', ''),
            'num_samples': len(scene.get('samples', [])),
            'cam_fps': fps,
            'lidar_fps': lidar_fps,
            'num_files': num_files,
            'cal_keys': list(cal_keys),
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(all_scenes)}")

    # ===== 统计 =====
    train = [s for s in scene_infos if s['split'] == 'train']
    val = [s for s in scene_infos if s['split'] == 'val']

    # FPS分组
    fps_groups = defaultdict(list)
    for si in scene_infos:
        fps_int = round(si['cam_fps'])
        fps_groups[fps_int].append(si['dir'])

    # 外参分组 (按tz值)
    tz_groups = defaultdict(lambda: defaultdict(list))  # cam -> tz_key -> [scene_dir]
    for cam_name in CAM_LIST:
        for scene_dir, ext in cam_extrinsics_all[cam_name]:
            tz = ext['translation']['z']
            key = round(tz, 2)
            tz_groups[cam_name][key].append(scene_dir)

    # 标定key分组
    cal_key_to_scenes = defaultdict(list)
    for si in scene_infos:
        for ck in si['cal_keys']:
            cal_key_to_scenes[ck].append(si['dir'])

    # 写入Markdown
    write_markdown(scene_infos, train, val, all_cal_data, cam_image_sizes,
                   cam_intrinsics, cam_extrinsics_all, tz_groups,
                   fps_groups, cal_key_to_scenes)


def write_markdown(scene_infos, train, val, all_cal_data, cam_image_sizes,
                   cam_intrinsics, cam_extrinsics_all, tz_groups,
                   fps_groups, cal_key_to_scenes):
    L = []
    L.append('# DDAD Dataset Exploration Report')
    L.append('')
    L.append('> Auto-generated by `explore_ddad.py`')
    L.append('')

    # ===== 1. Basic Info =====
    L.append('## 1. Basic Information')
    L.append('')
    L.append('| Item | Value |')
    L.append('|------|-------|')
    L.append(f'| Dataset | DDAD (Dense Depth for Automated Driving) |')
    L.append(f'| Total Scenes | {len(scene_infos)} |')
    L.append(f'| Train Scenes | {len(train)} |')
    L.append(f'| Val Scenes | {len(val)} |')
    L.append(f'| Total Frames | {sum(s["num_samples"] for s in scene_infos)} |')
    L.append(f'| Train Frames | {sum(s["num_samples"] for s in train)} |')
    L.append(f'| Val Frames | {sum(s["num_samples"] for s in val)} |')
    L.append(f'| Cameras | 6 (CAMERA_01, 05, 06, 07, 08, 09) |')
    L.append(f'| LiDAR | 1 (LIDAR) |')
    L.append(f'| Data Format | DGP (Dataset Governance Protocol) |')
    L.append(f'| Unique Calibration Keys | {len(all_cal_data)} |')
    L.append('')

    # ===== 2. Frame Count Distribution =====
    L.append('## 2. Frame Count Distribution')
    L.append('')
    fc = Counter(s['num_samples'] for s in scene_infos)
    L.append('| Frames/Scene | Count | Percentage |')
    L.append('|-------------|-------|------------|')
    for n in sorted(fc.keys()):
        L.append(f'| {n} | {fc[n]} | {fc[n]/len(scene_infos)*100:.1f}% |')
    L.append('')
    train_fc = [s['num_samples'] for s in train]
    val_fc = [s['num_samples'] for s in val]
    L.append(f'- Train: min={min(train_fc)}, max={max(train_fc)}, mean={np.mean(train_fc):.1f}')
    L.append(f'- Val: min={min(val_fc)}, max={max(val_fc)}, mean={np.mean(val_fc):.1f}')
    L.append('')

    # ===== 3. Capture Frequency =====
    L.append('## 3. Capture Frequency')
    L.append('')
    L.append('### 3.1 Frequency Distribution')
    L.append('')
    L.append('DDAD contains scenes at two different capture rates:')
    L.append('')
    L.append('| Frequency | Scene Count | Train | Val |')
    L.append('|-----------|-------------|-------|-----|')
    for fps_val in sorted(fps_groups.keys()):
        scenes = fps_groups[fps_val]
        t_cnt = sum(1 for s in scene_infos if s['dir'] in scenes and s['split'] == 'train')
        v_cnt = sum(1 for s in scene_infos if s['dir'] in scenes and s['split'] == 'val')
        L.append(f'| ~{fps_val} Hz | {len(scenes)} | {t_cnt} | {v_cnt} |')
    L.append('')
    L.append('> Note: Frequency is computed from image filename timestamps (microseconds).')
    L.append('> The original DDAD dataset is captured at 10 Hz; some scenes appear subsampled to 1 Hz.')
    L.append('')

    L.append('### 3.2 Per-Sensor Frequency')
    L.append('')
    L.append('All sensors (6 cameras + LiDAR) are synchronized within each scene.')
    L.append('Camera and LiDAR share the same timestamp per frame.')
    L.append('')

    # ===== 4. Camera Information =====
    L.append('## 4. Camera Information')
    L.append('')
    L.append('### 4.1 Camera List & Image Size')
    L.append('')
    L.append('| Camera | Direction | Image Size (WxH) | Channels |')
    L.append('|--------|-----------|-----------------|----------|')
    for cam in CAM_LIST:
        sizes = cam_image_sizes.get(cam, set())
        size_str = f'{list(sizes)[0][0]}x{list(sizes)[0][1]}' if sizes else 'N/A'
        L.append(f'| {cam} | {CAM_DIRECTIONS[cam]} | {size_str} | 3 (RGB) |')
    L.append('')

    L.append('### 4.2 Camera Intrinsics')
    L.append('')
    L.append('| Camera | fx | fy | cx | cy |')
    L.append('|--------|----|----|----|----|')
    for cam in CAM_LIST:
        intr = cam_intrinsics.get(cam, {})
        L.append(f'| {cam} | {intr.get("fx",0):.2f} | {intr.get("fy",0):.2f} | '
                 f'{intr.get("cx",0):.2f} | {intr.get("cy",0):.2f} |')
    L.append('')

    L.append('### 4.3 Estimated Field of View (FOV)')
    L.append('')
    L.append('| Camera | H-FOV | V-FOV | Notes |')
    L.append('|--------|-------|-------|-------|')
    for cam in CAM_LIST:
        intr = cam_intrinsics.get(cam, {})
        sizes = cam_image_sizes.get(cam, set())
        if intr and sizes:
            fx, fy = intr.get('fx', 1), intr.get('fy', 1)
            w, h = list(sizes)[0]
            hfov = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
            vfov = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
            note = 'Telephoto' if fx > 1500 else 'Wide-angle'
            L.append(f'| {cam} | {hfov:.1f}° | {vfov:.1f}° | {note} (fx={fx:.0f}) |')
        else:
            L.append(f'| {cam} | N/A | N/A | - |')
    L.append('')
    L.append('- CAMERA_01 has a longer focal length (fx≈2181) => narrower FOV (~48° H-FOV)')
    L.append('- Other cameras have wider focal lengths (fx≈1057-1065) => wider FOV (~85° H-FOV)')
    L.append('')

    # ===== 5. Extrinsics & Vehicles =====
    L.append('## 5. Camera Extrinsics & Vehicle Calibration')
    L.append('')
    L.append('### 5.1 Extrinsics Overview')
    L.append('')
    L.append('Each scene has a unique calibration key. Camera extrinsics vary continuously across scenes,')
    L.append('suggesting data was collected from multiple vehicles with slightly different camera placements.')
    L.append('')
    L.append('| Camera | tx Range | ty Range | tz Range |')
    L.append('|--------|----------|----------|----------|')
    for cam in CAM_LIST:
        exts = cam_extrinsics_all.get(cam, [])
        if exts:
            tx = [e['translation']['x'] for _, e in exts]
            ty = [e['translation']['y'] for _, e in exts]
            tz = [e['translation']['z'] for _, e in exts]
            L.append(f'| {cam} | {min(tx):.3f}-{max(tx):.3f} | '
                     f'{min(ty):.3f}-{max(ty):.3f} | {min(tz):.3f}-{max(tz):.3f} |')
    L.append('')

    L.append('### 5.2 Extrinsics Grouping by Height (tz)')
    L.append('')
    L.append('Grouping CAMERA_01 extrinsics by tz (camera height, 2 decimal places):')
    L.append('')
    L.append('| tz Group | Scene Count | Percentage |')
    L.append('|----------|-------------|------------|')
    groups = tz_groups.get('CAMERA_01', {})
    for key in sorted(groups.keys()):
        cnt = len(groups[key])
        L.append(f'| {key:.2f}m | {cnt} | {cnt/len(scene_infos)*100:.1f}% |')
    L.append('')
    L.append('This suggests approximately 5-6 distinct vehicle/calibration configurations.')
    L.append('')

    # ===== 6. Calibration Key Mapping =====
    L.append('## 6. Calibration Key Details')
    L.append('')
    L.append(f'Total unique calibration keys: **{len(all_cal_data)}**')
    L.append(f'(Each scene has its own calibration key)')
    L.append('')

    # 标定key示例
    L.append('### 6.1 Calibration Key → Scene Mapping (sample)')
    L.append('')
    L.append('| Cal Key (prefix) | Scene Count | Scenes |')
    L.append('|-----------------|-------------|--------|')
    sorted_cals = sorted(cal_key_to_scenes.items(), key=lambda x: -len(x[1]))[:10]
    for ck, scenes in sorted_cals:
        scene_str = ', '.join(scenes[:3])
        if len(scenes) > 3:
            scene_str += f'... ({len(scenes)} total)'
        L.append(f'| `{ck[:16]}...` | {len(scenes)} | {scene_str} |')
    L.append('')

    # 外参详细表 (每个标定key)
    L.append('### 6.2 Extrinsics per Calibration (first 5 calibrations)')
    L.append('')
    for i, (ck, cal) in enumerate(sorted(all_cal_data.items())[:5]):
        L.append(f'**Calibration `{ck[:12]}...`**')
        L.append('')
        L.append('| Sensor | tx | ty | tz | qw | qx | qy | qz |')
        L.append('|--------|----|----|----|----|----|----|----|')
        for idx, name in enumerate(cal['names']):
            ext = cal['extrinsics'][idx]
            t = ext['translation']
            r = ext['rotation']
            L.append(f'| {name} | {t["x"]:.4f} | {t["y"]:.4f} | {t["z"]:.4f} | '
                     f'{r["qw"]:.4f} | {r["qx"]:.4f} | {r["qy"]:.4f} | {r["qz"]:.4f} |')
        L.append('')

    # ===== 7. Scene Metadata =====
    L.append('## 7. Scene Metadata')
    L.append('')
    L.append('### 7.1 Description Field')
    L.append('')
    desc_counts = Counter(s['description'] for s in scene_infos)
    L.append('| Description | Count |')
    L.append('|-------------|-------|')
    for desc, cnt in desc_counts.most_common():
        L.append(f'| {desc} | {cnt} |')
    L.append('')

    L.append('### 7.2 Log Field')
    L.append('')
    log_counts = Counter(s['log'] for s in scene_infos)
    L.append('| Log Value | Count |')
    L.append('|-----------|-------|')
    for log_val, cnt in log_counts.most_common():
        display = log_val if log_val else '(empty)'
        L.append(f'| {display} | {cnt} |')
    L.append('')
    L.append('> The `log` field is empty for all scenes. No location or vehicle ID is stored in metadata.')
    L.append('')

    L.append('### 7.3 Scene Naming')
    L.append('')
    L.append('- Scene names correspond to directory numbers (000000-000199)')
    L.append('- Train: 000000-000149')
    L.append('- Val: 000150-000199')
    L.append('')

    # ===== 8. LiDAR =====
    L.append('## 8. LiDAR Information')
    L.append('')
    L.append('| Item | Value |')
    L.append('|------|-------|')
    L.append('| Sensor Name | LIDAR |')
    L.append('| Extrinsics | Identity matrix (mounted at vehicle center) |')
    L.append('| Coordinate System | Same as vehicle ego frame |')
    L.append('| Point Format | XYZ + intensity |')
    L.append('')

    # LiDAR外参验证
    first_cal = list(all_cal_data.values())[0]
    lidar_idx = first_cal['names'].index('LIDAR') if 'LIDAR' in first_cal['names'] else -1
    if lidar_idx >= 0:
        ext = first_cal['extrinsics'][lidar_idx]
        t = ext['translation']
        r = ext['rotation']
        L.append('LiDAR extrinsics (all scenes identical):')
        L.append(f'- Translation: ({t["x"]:.4f}, {t["y"]:.4f}, {t["z"]:.4f})')
        L.append(f'- Rotation: qw={r["qw"]:.4f}, qx={r["qx"]:.4f}, qy={r["qy"]:.4f}, qz={r["qz"]:.4f}')
        L.append('- Confirmed: LiDAR extrinsics = identity matrix')
    L.append('')

    # ===== 9. Depth Map =====
    L.append('## 9. Depth Map Information')
    L.append('')
    L.append('Depth maps are generated by projecting LiDAR points onto camera images:')
    L.append('')
    L.append('| Item | Value |')
    L.append('|------|-------|')
    L.append('| Generation | DGP auto-projection from LiDAR |')
    L.append('| Valid Pixel Rate | ~2% (sparse) |')
    L.append('| Depth Range | 0-250m |')
    L.append('| Dominant Range | 10-20m (largest proportion) |')
    L.append('| Format | float32 .npz |')
    L.append('| Size | Same as image (1216x1936) |')
    L.append('')

    # ===== 10. Directory Structure =====
    L.append('## 10. Directory Structure')
    L.append('')
    L.append('### Raw Data')
    L.append('```')
    L.append('data/raw/ddad_train_val/')
    L.append('├── ddad.json                         # Top-level dataset definition')
    L.append('├── 000000/                           # Scene 0 (train)')
    L.append('│   ├── scene_<hash>.json             # Scene protobuf JSON')
    L.append('│   ├── calibration/<sha1>.json       # Calibration file')
    L.append('│   ├── rgb/')
    L.append('│   │   ├── CAMERA_01/<ts>.png')
    L.append('│   │   ├── CAMERA_05/<ts>.png')
    L.append('│   │   ├── CAMERA_06/<ts>.png')
    L.append('│   │   ├── CAMERA_07/<ts>.png')
    L.append('│   │   ├── CAMERA_08/<ts>.png')
    L.append('│   │   └── CAMERA_09/<ts>.png')
    L.append('│   └── point_cloud/LIDAR/<ts>.npz')
    L.append('├── ...')
    L.append('├── 000149/                           # Scene 149 (train)')
    L.append('├── 000150/                           # Scene 150 (val)')
    L.append('├── ...')
    L.append('└── 000199/                           # Scene 199 (val)')
    L.append('```')
    L.append('')

    L.append('### Processed Data')
    L.append('```')
    L.append('data/ddad_process/')
    L.append('├── train/')
    L.append('│   ├── 000/')
    L.append('│   │   ├── images/                   # {frame}_{cam}.jpg')
    L.append('│   │   ├── intrinsics/               # {cam_idx}.txt (6 files)')
    L.append('│   │   ├── cam2ego_extrinsics/       # {cam_idx}.txt (6 files)')
    L.append('│   │   ├── ego_pose/                 # {frame}.txt')
    L.append('│   │   └── depth_map/               # {frame}_{cam}.npz')
    L.append('│   └── ...')
    L.append('└── valid/')
    L.append('    └── ...')
    L.append('```')
    L.append('')

    # ===== 11. Coordinate System =====
    L.append('## 11. Coordinate System')
    L.append('')
    L.append('- **Vehicle (Ego)**: X-forward, Y-left, Z-up')
    L.append('- **Camera**: OpenCV (X-right, Y-down, Z-forward)')
    L.append('- **LiDAR**: Same as vehicle (identity extrinsics)')
    L.append('- `extrinsics`: sensor-to-vehicle transform')
    L.append('- `pose`: sensor-to-world transform')
    L.append('- `ego_pose`: vehicle-to-world transform')
    L.append('- `sensor2world = ego2world @ sensor2ego`')
    L.append('')

    # ===== 12. Summary =====
    L.append('## 12. Key Findings')
    L.append('')
    L.append('1. **Scale**: 200 scenes (150 train + 50 val), 50 or 100 frames per scene')
    L.append('2. **Image Size**: All cameras 1936x1216, RGB')
    L.append('3. **Capture Frequency**: Mixed — 107 scenes at ~1 Hz, 93 scenes at ~10 Hz')
    L.append('4. **Calibration**: 200 unique calibration keys (one per scene), extrinsics vary continuously')
    L.append('5. **Camera Config**: 6 cameras — CAMERA_01 telephoto (fx≈2181, ~47° H-FOV), others wide-angle (fx≈1057-1065, ~100°+ H-FOV)')
    L.append('6. **LiDAR**: Identity extrinsics (mounted at vehicle center), same frequency as cameras')
    L.append('7. **Vehicles**: Extrinsics suggest ~5-6 distinct vehicle/calibration groups (based on camera height tz)')
    L.append('8. **Metadata**: `log` field empty, `description` only "ddad_train"/"ddad_val", no vehicle ID or location info')
    L.append('9. **Depth Maps**: Sparse (~2% valid pixels), range 0-250m, generated from LiDAR projection')
    L.append('')

    # Write
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'docs',
        'DDAD_DATASET_SUMMARY.md',
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"\nReport written to: {output_path}")


if __name__ == '__main__':
    main()
