"""
batch_merge_depthmap_nuscenes.py — 批量处理 nuScenes 的 depth map 多帧合并 + 遮挡过滤

基于 tmp_multilidar_merge_v3.py 的核心逻辑，批量处理所有场景的所有帧。
使用各相机的 depth map 反投影得到点云（而非 lidar bin），通过 extrinsics 变换。
只保存指定相机的深度图 npz，不保存可视化。

用法：直接修改下方全局变量，然后 python batch_merge_depthmap_nuscenes.py

输出：
    {OUT_DIR}/{scene}/depth_map/{frame:03d}_{cam_id}.npz
"""

import os
import glob
import json
import numpy as np
import open3d as o3d
import time
from tqdm import tqdm


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  全局配置                                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

DATA_ROOT   = 'data/nuscenes/processed_10Hz/trainval'
SCENE_LIST = './nuScenes_Val2.txt'
OUT_DIR     = 'data/nuscenes/processed_10Hz/trainval_multi_depthmap'
CAM_LIST    = [5, 4, 3, 2, 1, 0]    # 用于构建体素网格的相机列表
RENDER_CAM_LIST = [5]                # 只保存这些相机的深度图
N_BEFORE    = 8                      # 向前合并帧数上限
N_AFTER     = 8                      # 向后合并帧数上限
FILTER_ALL  = False
SELF_RANGE_DEPTH = 1.5               # depth map 深度过滤阈值 (m)
START_FRAME = 18                      # 从第几帧开始处理

# 体素参数
VOXEL_SIZE  = 0.15
VOXEL_ALL   = False

# bbox 参数
BBOX_VOXEL  = True
BBOX_EXPAND = 1.1
BBOX_DILATE = True
MAXMUM_FILTER_SIZE = 3

MERGE_TARGET = True


# ──────────────────────────────────────────────────────────────────────
# I/O 工具
# ──────────────────────────────────────────────────────────────────────

def load_txt_matrix(path, shape=(4, 4)):
    return np.loadtxt(path, dtype=np.float64).reshape(shape)


def load_intrinsics(path):
    vals = np.loadtxt(path, dtype=np.float64)
    if len(vals) == 9 and vals[4:].sum() == 0:
        fx, fy, cx, cy = vals[0], vals[1], vals[2], vals[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return vals.reshape(3, 3)


def load_depth_map(path):
    data = np.load(path)
    return data['depth']


def load_instances(instances_dir):
    with open(os.path.join(instances_dir, 'instances_info.json'), 'r') as f:
        return json.load(f)


def load_frame_instances(instances_dir):
    with open(os.path.join(instances_dir, 'frame_instances.json'), 'r') as f:
        return json.load(f)


def get_all_frame_indices(scene_dir):
    dm_dir = os.path.join(scene_dir, 'depth_map')
    if not os.path.exists(dm_dir):
        return []
    files = glob.glob(os.path.join(dm_dir, '*.npz'))
    indices = sorted(set(int(os.path.basename(f).split('_')[0]) for f in files))
    return indices


# ──────────────────────────────────────────────────────────────────────
# 核心计算
# ──────────────────────────────────────────────────────────────────────

def depth_to_camera_points(depth_map, K):
    K_inv = np.linalg.inv(K)
    H, W = depth_map.shape
    v_coords, u_coords = np.where(depth_map > SELF_RANGE_DEPTH)
    depths = depth_map[v_coords, u_coords]
    pixels_h = np.stack([u_coords, v_coords, np.ones_like(u_coords)], axis=0).astype(np.float64)
    points_cam = (K_inv @ pixels_h) * depths[np.newaxis, :]
    points_cam = points_cam.T
    return np.concatenate([points_cam, depths[:, np.newaxis]], axis=1).astype(np.float32)


def camera_to_world(points, c2w):
    R, t = c2w[:3, :3], c2w[:3, 3]
    xyz = (R @ points[:, :3].T).T + t
    return np.concatenate([xyz, points[:, 3:]], axis=1)


def world_to_camera(points, c2w):
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    xyz = (R @ points[:, :3].T).T + t
    return np.concatenate([xyz, points[:, 3:]], axis=1)


def filter_dynamic_objects(points_world, instances_info, frame_instances, frame_idx):
    frame_key = str(frame_idx)
    if frame_key not in frame_instances:
        return np.ones(len(points_world), dtype=bool)
    ids = frame_instances[frame_key]
    if not ids:
        return np.ones(len(points_world), dtype=bool)
    pts = points_world[:, :3]
    mask = np.ones(len(pts), dtype=bool)
    for inst_id in ids:
        inst_key = str(inst_id)
        if inst_key not in instances_info:
            continue
        inst = instances_info[inst_key]
        fi_list = inst['frame_annotations']['frame_idx']
        if frame_idx not in fi_list:
            continue
        fi = fi_list.index(frame_idx)
        center = np.array(inst['frame_annotations']['obj_to_world'][fi])[:3, 3].copy()
        half = np.array(inst['frame_annotations']['box_size'][fi]) * BBOX_EXPAND / 2.0
        mask &= ~np.all(np.abs(pts - center) < half, axis=1)
    return mask


def points_to_depth_map(points_cam, K, W, H):
    z = points_cam[:, 2]
    valid = z > 0
    pts = points_cam[valid]
    proj = (K @ pts[:, :3].T).T
    u = (proj[:, 0] / proj[:, 2]).astype(np.int32)
    v = (proj[:, 1] / proj[:, 2]).astype(np.int32)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pts, u, v = pts[in_img], u[in_img], v[in_img]
    dm = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(dm, (v, u), pts[:, 2])
    dm[dm == np.inf] = 0.0
    return dm


# ──────────────────────────────────────────────────────────────────────
# 体素占用网格
# ──────────────────────────────────────────────────────────────────────

class VoxelOccupancyGrid:

    def __init__(self, points_world, voxel_size=0.5):
        self.voxel_size = voxel_size
        xyz = points_world[:, :3].astype(np.float64)
        self.min_bound = xyz.min(axis=0)
        self.max_bound = xyz.max(axis=0)
        indices = np.floor((xyz - self.min_bound) / voxel_size).astype(np.int32)
        self.grid_shape = indices.max(axis=0) + 1
        n_total = np.prod(self.grid_shape)
        self.grid = np.zeros(self.grid_shape, dtype=np.uint8)
        self.grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        n_occupied = self.grid.sum()
        self._build_scene()

    def add_bboxes(self, bboxes):
        grid_before = self.grid.copy()
        for bbox in bboxes:
            center = bbox[:3]
            half = bbox[3:6] / 2.0
            bbox_min = center - half
            bbox_max = center + half
            idx_min = np.maximum(np.floor((bbox_min - self.min_bound) / self.voxel_size).astype(np.int32), 0)
            idx_max = np.minimum(np.floor((bbox_max - self.min_bound) / self.voxel_size).astype(np.int32),
                                 np.array(self.grid_shape) - 1)
            self.grid[idx_min[0]:idx_max[0]+1,
                      idx_min[1]:idx_max[1]+1,
                      idx_min[2]:idx_max[2]+1] = 1
        if BBOX_DILATE:
            from scipy.ndimage import maximum_filter
            bbox_only = (self.grid > 0) & (grid_before == 0)
            bbox_dilated = maximum_filter(bbox_only.astype(np.uint8), size=MAXMUM_FILTER_SIZE).astype(np.uint8)
            self.grid = np.maximum(self.grid, bbox_dilated)
        self._build_scene()

    def _build_scene(self):
        occupied_idx = np.argwhere(self.grid > 0).astype(np.float64)
        n_occ = len(occupied_idx)
        if n_occ == 0:
            self.scene = o3d.t.geometry.RaycastingScene()
            return
        voxel_corners = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]
        ], dtype=np.float64)
        cube_faces = np.array([
            [0,1,2],[0,2,3],[4,6,5],[4,7,6],
            [0,4,5],[0,5,1],[1,5,6],[1,6,2],
            [2,6,7],[2,7,3],[3,7,4],[3,4,0],
        ], dtype=np.int32)
        all_verts = (occupied_idx[:, None, :] + voxel_corners[None, :, :])
        all_verts = (all_verts * self.voxel_size + self.min_bound).reshape(-1, 3)
        offsets = np.arange(n_occ, dtype=np.int32) * 8
        all_faces = (cube_faces[None, :, :] + offsets[:, None, None]).reshape(-1, 3)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(all_verts)
        mesh.triangles = o3d.utility.Vector3iVector(all_faces)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def batch_raycast(self, origins, directions, max_dist=100.0):
        rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = ans['t_hit'].numpy()
        t_hit[t_hit == np.inf] = max_dist
        return t_hit


def filter_occluded_by_voxels(voxel_grid, points_cam, cam2world, max_dist=100.0):
    cam_origin_world = cam2world[:3, 3]
    R_c2w = cam2world[:3, :3]
    pts_cam = points_cam[:, :3].astype(np.float32)
    dists = np.linalg.norm(pts_cam, axis=1)
    dirs_cam = pts_cam / dists[:, np.newaxis]
    dirs_world = (R_c2w @ dirs_cam.T).astype(np.float32).T
    origins = np.tile(cam_origin_world.astype(np.float32), (len(pts_cam), 1))
    t_hits = voxel_grid.batch_raycast(origins, dirs_world, max_dist=max_dist)
    visible = (t_hits >= dists - voxel_grid.voxel_size)
    return visible


# ──────────────────────────────────────────────────────────────────────
# 辅助：加载某帧某相机的 depth map → 世界坐标点云
# ──────────────────────────────────────────────────────────────────────

def load_frame_cam_to_world(scene_dir, frame_idx, cam_id):
    dm_path = os.path.join(scene_dir, 'depth_map', f'{frame_idx:03d}_{cam_id}.npz')
    if not os.path.exists(dm_path):
        return None
    depth = load_depth_map(dm_path)
    K = load_intrinsics(os.path.join(scene_dir, 'intrinsics', f'{cam_id}.txt'))
    c2w = load_txt_matrix(os.path.join(scene_dir, 'extrinsics', f'{frame_idx:03d}_{cam_id}.txt'))
    cam_pts = depth_to_camera_points(depth, K)
    world_pts = camera_to_world(cam_pts, c2w)
    return world_pts


# ──────────────────────────────────────────────────────────────────────
# 处理单帧
# ──────────────────────────────────────────────────────────────────────

def process_single_frame(scene_dir, target_frame, all_indices, instances_info, frame_instances, cam_id, K, H, W):
    """处理单帧，返回 depth map [H, W] 或 None"""
    t0 = time.time()

    frame_start = max(0, target_frame - N_BEFORE)
    frame_end = min(all_indices[-1] + 1, target_frame + N_AFTER + 1)
    supp_frames = [f for f in range(frame_start, frame_end) if f != target_frame]

    # ── 构建体素网格（使用 CAM_LIST 所有相机的 depth map）──
    voxel_frame_indices = supp_frames + [target_frame] if not VOXEL_ALL else all_indices

    voxel_world_list = []
    for f in voxel_frame_indices:
        for cid in CAM_LIST:
            world_pts = load_frame_cam_to_world(scene_dir, f, cid)
            if world_pts is None:
                continue
            if f != target_frame or not MERGE_TARGET:
                dyn_mask = filter_dynamic_objects(world_pts, instances_info, frame_instances, f)
                world_pts = world_pts[dyn_mask]
            voxel_world_list.append(world_pts[:, :3])

    if not voxel_world_list:
        return None

    voxel_all_points = np.concatenate(voxel_world_list, axis=0)
    t1 = time.time()

    voxel_grid = VoxelOccupancyGrid(voxel_all_points, voxel_size=VOXEL_SIZE)
    t2 = time.time()

    # bbox 填充
    if BBOX_VOXEL:
        frame_key = str(target_frame)
        if frame_key in frame_instances:
            bboxes = []
            for inst_id in frame_instances[frame_key]:
                inst_key = str(inst_id)
                if inst_key not in instances_info:
                    continue
                inst = instances_info[inst_key]
                fi_list = inst['frame_annotations']['frame_idx']
                if target_frame not in fi_list:
                    continue
                fi = fi_list.index(target_frame)
                box_size = np.array(inst['frame_annotations']['box_size'][fi]) * BBOX_EXPAND
                obj2world = np.array(inst['frame_annotations']['obj_to_world'][fi])
                center = obj2world[:3, 3]
                bboxes.append(np.concatenate([center, box_size]))
            if bboxes:
                voxel_grid.add_bboxes(np.array(bboxes))

    # ── 加载目标帧 c2w ──
    c2w = load_txt_matrix(os.path.join(scene_dir, 'extrinsics', f'{target_frame:03d}_{cam_id}.txt'))

    # ── 补充帧：遮挡过滤 ──
    supp_cam_list = []
    for f in supp_frames:
        world_pts = load_frame_cam_to_world(scene_dir, f, cam_id)
        if world_pts is None:
            continue
        dyn_mask = filter_dynamic_objects(world_pts, instances_info, frame_instances, f)
        world_pts = world_pts[dyn_mask]
        cam_pts = world_to_camera(world_pts, c2w)
        supp_cam_list.append(cam_pts)

    supp_cam = np.concatenate(supp_cam_list, axis=0) if supp_cam_list else np.empty((0, 4))
    t3 = time.time()
    supp_visible = filter_occluded_by_voxels(voxel_grid, supp_cam, c2w)
    supp_filtered = supp_cam[supp_visible]
    t4 = time.time()

    # ── 合并目标帧 + 补充帧 ──
    if MERGE_TARGET:
        target_world_pts = load_frame_cam_to_world(scene_dir, target_frame, cam_id)
        if target_world_pts is not None:
            target_cam = world_to_camera(target_world_pts, c2w)
        else:
            target_cam = np.empty((0, 4))
        vis_cam = np.concatenate([target_cam, supp_filtered], axis=0)
    else:
        vis_cam = supp_filtered

    if len(vis_cam) == 0:
        return np.zeros((H, W), dtype=np.float32)

    depth_map = points_to_depth_map(vis_cam, K, W, H)
    t5 = time.time()
    # print(f'    帧{target_frame:03d}_cam{cam_id}: 体素{t2-t1:.1f}s 补充帧{t3-t2:.1f}s '
    #       f'过滤{t4-t3:.1f}s 投影{t5-t4:.1f}s 合计{t5-t0:.1f}s')
    return depth_map


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────

def main():
    with open(SCENE_LIST, 'r') as f:
        scene_names = [line.strip() for line in f if line.strip()]

    print(f'[INFO] 总场景数: {len(scene_names)}')
    print(f'[INFO] 输出目录: {OUT_DIR}')
    print(f'[INFO] 体素相机: {CAM_LIST}, 保存相机: {RENDER_CAM_LIST}')
    print(f'[INFO] N_BEFORE={N_BEFORE}, N_AFTER={N_AFTER}, VOXEL_SIZE={VOXEL_SIZE}')

    total_start = time.time()

    for scene_idx, scene_name in enumerate(tqdm(scene_names, desc='场景')):
        scene_dir = os.path.join(DATA_ROOT, scene_name)
        if not os.path.exists(scene_dir):
            continue

        out_scene_dir = os.path.join(OUT_DIR, scene_name, 'depth_map')
        os.makedirs(out_scene_dir, exist_ok=True)

        all_indices = get_all_frame_indices(scene_dir)
        if not all_indices:
            continue

        instances_dir = os.path.join(scene_dir, 'instances')
        if not os.path.exists(instances_dir):
            continue
        instances_info = load_instances(instances_dir)
        frame_instances = load_frame_instances(instances_dir)

        # 预加载每个相机的 intrinsics 和图像尺寸
        cam_params = {}
        for cam_id in RENDER_CAM_LIST:
            K = load_intrinsics(os.path.join(scene_dir, 'intrinsics', f'{cam_id}.txt'))
            ref_dm = os.path.join(scene_dir, 'depth_map', f'{all_indices[0]:03d}_{cam_id}.npz')
            if os.path.exists(ref_dm):
                H, W = np.load(ref_dm)['depth'].shape
            else:
                H, W = 900, 1600
            cam_params[cam_id] = {'K': K, 'H': H, 'W': W}

        scene_start = time.time()
        n_done = 0
        n_skip = 0
        frame_indices = [f for f in all_indices if f >= START_FRAME]

        for frame_idx in tqdm(frame_indices, desc=f"帧 {scene_name}", leave=False):
            for cam_id in RENDER_CAM_LIST:
                out_path = os.path.join(out_scene_dir, f'{frame_idx:03d}_{cam_id}.npz')
                if os.path.exists(out_path):
                    n_skip += 1
                    continue

                cp = cam_params[cam_id]
                depth_map = process_single_frame(
                    scene_dir, frame_idx, all_indices,
                    instances_info, frame_instances,
                    cam_id, cp['K'], cp['H'], cp['W']
                )

                if depth_map is None:
                    continue

                cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
                np.savez_compressed(
                    out_path,
                    depth=depth_map,
                    cam_name=f'CAM_{cam_names[cam_id]}',
                    cam_id=np.int32(cam_id),
                )
                n_done += 1

        scene_time = time.time() - scene_start
        print(f'[INFO] 场景 {scene_idx+1}/{len(scene_names)} {scene_name}: '
              f'处理 {n_done} 帧, 跳过 {n_skip} 帧, 耗时 {scene_time:.1f}s')

    total_time = time.time() - total_start
    print(f'\n[INFO] 全部完成！总耗时: {total_time/3600:.1f}h')


if __name__ == '__main__':
    main()
