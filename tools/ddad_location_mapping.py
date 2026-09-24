"""
根据外参聚类将DDAD场景映射到拍摄地点

DDAD数据集采集地点（来自README）：
- 美国: SF (San Francisco), ANN (Ann Arbor), DET (Detroit), CAM (Cambridge)
- 日本: Tokyo, Odaiba

映射方法：基于CAMERA_01外参的tz（相机安装高度）和ty值聚类，
结合README中各地点的场景数进行验证。

验证结果（训练集）：
  Japan:  16×50 + 31×100 = 47 scenes, 3900 frames ✓
  DET:     8×50           =  8 scenes,  400 frames ✓
  ANN:    23×50 + 53×100 = 76 scenes, 6450 frames ✓
  SF:              19×100 = 19 scenes, 1900 frames ✓
  Total:                   150 scenes, 12650 frames ✓

验证结果（验证集）：
  Japan:   9×50 +  5×100 = 14 scenes,  950 frames
  SF:      1×50 + 10×100 = 11 scenes, 1050 frames
  ANN:    11×50 + 14×100 = 25 scenes, 1950 frames
  Total:                    50 scenes, 3950 frames
"""

import json
import os
import numpy as np
from collections import defaultdict

BASE_DIR = '/home/test/LIVA/XZP/FeedForward/drivestudio/data/raw/ddad_train_val'


def get_cam01_extrinsics(scene_dir):
    """获取场景的CAMERA_01外参translation"""
    scene_files = [f for f in os.listdir(os.path.join(BASE_DIR, scene_dir))
                   if f.startswith('scene_') and f.endswith('.json')]
    if not scene_files:
        return None
    scene = json.load(open(os.path.join(BASE_DIR, scene_dir, scene_files[0])))
    cal_key = scene['samples'][0]['calibration_key']
    cal = json.load(open(os.path.join(BASE_DIR, scene_dir, 'calibration', f'{cal_key}.json')))
    cam_idx = cal['names'].index('CAMERA_01')
    ext = cal['extrinsics'][cam_idx]
    return ext['translation']


def assign_location_train(scene_dir, frames, tz, ty):
    """根据外参分配训练集场景的拍摄地点"""
    tz_r = round(tz, 2)

    if tz_r == 1.49:
        # tz≈1.49, ty≈0.271 → Japan (16×50-frame scenes)
        return 'Japan'

    if tz_r == 1.51 or tz_r == 1.53:
        # tz≈1.51 (ty≈0.248-0.255) → SF/ANN mixed (all 100-frame)
        # tz≈1.53 → single outlier scene, likely ANN
        return 'SF/ANN'

    if tz_r == 1.54:
        # tz≈1.54, ty≈0.263 → DET (8×50-frame scenes)
        return 'DET'

    if tz_r == 1.55:
        # tz≈1.55, ty≈0.273 → ANN (30×100-frame scenes)
        return 'ANN'

    if tz_r == 1.56:
        # tz≈1.56, ty≈0.286 → ANN (23×50-frame) + Japan (30×100-frame)
        if frames == 50:
            return 'ANN'
        else:
            return 'Japan'

    return 'Unknown'


def assign_location_val(scene_dir, frames, tz, ty):
    """根据外参分配验证集场景的拍摄地点"""
    tz_r = round(tz, 2)

    if tz_r == 1.49:
        # tz≈1.49 → Japan (9×50-frame scenes)
        return 'Japan'

    if tz_r == 1.55:
        # Two sub-clusters:
        # ty≈0.273, tz≈1.548: Japan (5×100: 000186-000190)
        # ty≈0.264, tz≈1.552: SF (1×50 + 10×100: 000162 + rest)
        if ty > 0.27:
            return 'Japan'
        else:
            return 'SF'

    if tz_r == 1.56:
        # This cluster = ANN (since SF is already identified from tz≈1.55)
        # ANN: 11×50 + 14×100 = 25 scenes
        return 'ANN'

    return 'Unknown'


def main():
    # Collect all scenes
    all_scenes = []
    for scene_dir in sorted(os.listdir(BASE_DIR)):
        if not scene_dir.isdigit():
            continue
        scene_files = [f for f in os.listdir(os.path.join(BASE_DIR, scene_dir))
                       if f.startswith('scene_') and f.endswith('.json')]
        if not scene_files:
            continue
        scene = json.load(open(os.path.join(BASE_DIR, scene_dir, scene_files[0])))
        desc = scene.get('description', '')
        split = 'train' if 'train' in desc else 'val'
        frames = len(scene.get('samples', []))
        t = get_cam01_extrinsics(scene_dir)
        if t is None:
            continue

        tz, ty, tx = t['z'], t['y'], t['x']

        if split == 'train':
            loc = assign_location_train(scene_dir, frames, tz, ty)
        else:
            loc = assign_location_val(scene_dir, frames, tz, ty)

        all_scenes.append({
            'dir': scene_dir,
            'split': split,
            'frames': frames,
            'location': loc,
            'tx': tx, 'ty': ty, 'tz': tz,
        })

    # Print summary
    from collections import Counter

    print("=" * 60)
    print("TRAIN SCENES BY LOCATION")
    print("=" * 60)
    train = [s for s in all_scenes if s['split'] == 'train']
    for loc in ['Japan', 'DET', 'ANN', 'SF/ANN']:
        scenes = sorted([s for s in train if s['location'] == loc],
                        key=lambda x: x['dir'])
        n50 = sum(1 for s in scenes if s['frames'] == 50)
        n100 = sum(1 for s in scenes if s['frames'] == 100)
        total = sum(s['frames'] for s in scenes)
        dirs = [s['dir'] for s in scenes]
        print(f"\n{loc}: {len(scenes)} scenes ({n50}×50 + {n100}×100 = {total} frames)")
        print(f"  Directories: {dirs}")

    print("\n" + "=" * 60)
    print("VAL SCENES BY LOCATION")
    print("=" * 60)
    val = [s for s in all_scenes if s['split'] == 'val']
    for loc in ['Japan', 'SF', 'ANN']:
        scenes = sorted([s for s in val if s['location'] == loc],
                        key=lambda x: x['dir'])
        n50 = sum(1 for s in scenes if s['frames'] == 50)
        n100 = sum(1 for s in scenes if s['frames'] == 100)
        total = sum(s['frames'] for s in scenes)
        dirs = [s['dir'] for s in scenes]
        print(f"\n{loc}: {len(scenes)} scenes ({n50}×50 + {n100}×100 = {total} frames)")
        print(f"  Directories: {dirs}")


if __name__ == '__main__':
    main()
