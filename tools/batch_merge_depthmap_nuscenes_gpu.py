"""
batch_merge_depthmap_nuscenes_gpu.py — GPU 加速版批量 depth map 合并 + 遮挡过滤

基于 batch_merge_depthmap_nuscenes.py，使用 PyTorch GPU 加速：
- 点云变换（world_to_camera）在 GPU 上批量计算
- 动态物体过滤在 GPU 上向量化（无 Python 循环）
- 体素网格构建在 CPU（Open3D RaycastingScene，C++ 后端已足够快）

用法：直接修改下方全局变量，然后 python tools/batch_merge_depthmap_nuscenes_gpu.py
"""

import os
import glob
import json
import numpy as np
import open3d as o3d
import torch
import time
from tqdm import tqdm


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  全局配置                                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

DATA_ROOT   = 'data/nuscenes/processed_10Hz/trainval'
SCENE_LIST = './nuScenes_Val2.txt'
OUT_DIR     = 'data/nuscenes/processed_10Hz/trainval_multi_depthmap'
CAM_LIST    = [5, 4, 3, 2, 1, 0]
RENDER_CAM_LIST = [5]
N_BEFORE    = 8
N_AFTER     = 8
FILTER_ALL  = False
SELF_RANGE_DEPTH = 1.5
START_FRAME = 35

VOXEL_SIZE  = 0.15
DEPTH_MAX   = 110.0
VOXEL_ALL   = False

BBOX_VOXEL  = True
BBOX_EXPAND = 1.1
BBOX_DILATE = True
MAXMUM_FILTER_SIZE = 3

MERGE_TARGET = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    return sorted(set(int(os.path.basename(f).split('_')[0]) for f in files))


# ──────────────────────────────────────────────────────────────────────
# GPU 核心计算
# ──────────────────────────────────────────────────────────────────────

def depth_to_camera_points_gpu(depth_map, K_inv_t):
    """depth map → 相机坐标系 3D 点（GPU）"""
    H, W = depth_map.shape
    v_coords, u_coords = np.where(depth_map > SELF_RANGE_DEPTH)
    if len(v_coords) == 0:
        return torch.empty(0, 4, device=DEVICE)
    depths = depth_map[v_coords, u_coords]

    pixels_h = np.stack([u_coords, v_coords, np.ones_like(u_coords, dtype=np.float64)], axis=0)
    # GPU 计算: K_inv @ pixels * depths
    pixels_t = torch.from_numpy(pixels_h).float().to(DEVICE)       # [3, N]
    depths_t = torch.from_numpy(depths).float().to(DEVICE)         # [N]
    pts_cam = (K_inv_t @ pixels_t) * depths_t[None, :]             # [3, N]
    pts_cam = pts_cam.T                                            # [N, 3]
    return torch.cat([pts_cam, depths_t[:, None]], dim=1)          # [N, 4]


def camera_to_world_gpu(pts, c2w_t):
    """GPU 批量变换: 相机坐标 → 世界坐标"""
    R = c2w_t[:3, :3]   # [3, 3]
    t = c2w_t[:3, 3]    # [3]
    xyz = (R @ pts[:, :3].T).T + t  # [N, 3]
    return torch.cat([xyz, pts[:, 3:]], dim=1)


def world_to_camera_gpu(pts, w2c_t):
    """GPU 批量变换: 世界坐标 → 相机坐标（接收预计算的 w2c）"""
    R = w2c_t[:3, :3]
    t = w2c_t[:3, 3]
    xyz = (R @ pts[:, :3].T).T + t
    return torch.cat([xyz, pts[:, 3:]], dim=1)


def filter_dynamic_objects_cpu(pts_np, bbox_cache, frame_idx):
    """CPU 循环过滤（避免 GPU 大张量分配）"""
    if frame_idx not in bbox_cache:
        return np.ones(len(pts_np), dtype=bool)
    centers_t, halves_t = bbox_cache[frame_idx]
    centers = centers_t.cpu().numpy()
    halves = halves_t.cpu().numpy()
    pts = pts_np[:, :3]
    mask = np.ones(len(pts), dtype=bool)
    for j in range(len(centers)):
        mask &= ~np.all(np.abs(pts - centers[j]) < halves[j], axis=1)
    return mask


def precompute_bboxes_gpu(instances_info, frame_instances, all_indices):
    """预计算所有帧的 bbox（GPU tensors）"""
    bbox_cache = {}
    for f_idx in all_indices:
        frame_key = str(f_idx)
        if frame_key not in frame_instances or not frame_instances[frame_key]:
            continue
        centers, halves = [], []
        for inst_id in frame_instances[frame_key]:
            inst_key = str(inst_id)
            if inst_key not in instances_info:
                continue
            inst = instances_info[inst_key]
            fi_list = inst['frame_annotations']['frame_idx']
            if f_idx not in fi_list:
                continue
            fi = fi_list.index(f_idx)
            center = np.array(inst['frame_annotations']['obj_to_world'][fi])[:3, 3]
            half = np.array(inst['frame_annotations']['box_size'][fi]) * BBOX_EXPAND / 2.0
            centers.append(center)
            halves.append(half)
        if centers:
            bbox_cache[f_idx] = (
                torch.tensor(np.array(centers), dtype=torch.float32, device=DEVICE),
                torch.tensor(np.array(halves), dtype=torch.float32, device=DEVICE),
            )
    return bbox_cache


def points_to_depth_map_gpu(pts_cam_t, K_t, W, H):
    """GPU 投影: 相机坐标系 3D 点 → 深度图 [H, W]"""
    if len(pts_cam_t) == 0:
        return np.zeros((H, W), dtype=np.float32)

    z = pts_cam_t[:, 2]
    valid = z > 0
    pts = pts_cam_t[valid]

    proj = (K_t @ pts[:, :3].T).T  # [N, 3]
    u = (proj[:, 0] / proj[:, 2]).long()
    v = (proj[:, 1] / proj[:, 2]).long()

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pts = pts[in_img]
    u = u[in_img]
    v = v[in_img]

    # GPU scatter 取最小深度
    depth_map = torch.full((H, W), float('inf'), device=DEVICE)
    depth_map[v, u] = torch.min(depth_map[v, u], pts[:, 2])
    depth_map[depth_map == float('inf')] = 0.0
    return depth_map.cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# 体素占用网格（CPU，Open3D C++ 后端）
# ──────────────────────────────────────────────────────────────────────

class VoxelOccupancyGrid:
    """GPU 加速体素占用网格，使用 PyTorch 实现 Amanatides & Woo 光线遍历"""

    def __init__(self, points_world_np, voxel_size=0.5):
        self.voxel_size = voxel_size
        xyz = points_world_np[:, :3].astype(np.float64)
        self.min_bound = xyz.min(axis=0)
        self.max_bound = xyz.max(axis=0)
        indices = np.floor((xyz - self.min_bound) / voxel_size).astype(np.int32)
        self.grid_shape = indices.max(axis=0) + 1
        self.grid = np.zeros(self.grid_shape, dtype=np.uint8)
        self.grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        # GPU tensor 用于光线遍历
        self.grid_t = torch.from_numpy(self.grid).to(DEVICE)
        self.min_bound_t = torch.from_numpy(self.min_bound).float().to(DEVICE)

    def add_bboxes(self, bboxes_np):
        grid_before = self.grid.copy()
        for bbox in bboxes_np:
            center, half = bbox[:3], bbox[3:6] / 2.0
            idx_min = np.maximum(np.floor((center - half - self.min_bound) / self.voxel_size).astype(np.int32), 0)
            idx_max = np.minimum(np.floor((center + half - self.min_bound) / self.voxel_size).astype(np.int32),
                                 np.array(self.grid_shape) - 1)
            self.grid[idx_min[0]:idx_max[0]+1, idx_min[1]:idx_max[1]+1, idx_min[2]:idx_max[2]+1] = 1
        if BBOX_DILATE:
            from scipy.ndimage import maximum_filter
            bbox_only = (self.grid > 0) & (grid_before == 0)
            bbox_dilated = maximum_filter(bbox_only.astype(np.uint8), size=MAXMUM_FILTER_SIZE).astype(np.uint8)
            self.grid = np.maximum(self.grid, bbox_dilated)
        self.grid_t = torch.from_numpy(self.grid).to(DEVICE)

    def batch_raycast(self, origins_np, directions_np, max_dist=100.0):
        """
        GPU 向量化 Amanatides & Woo 体素遍历

        origins_np: [N, 3] float32 世界坐标光线起点
        directions_np: [N, 3] float32 世界坐标光线方向（已归一化）
        返回: [N] float32 每条光线到第一个占用体素的距离
        """
        origins_t = torch.from_numpy(origins_np).float().to(DEVICE)    # [N, 3]
        dirs_t = torch.from_numpy(directions_np).float().to(DEVICE)    # [N, 3]
        N = len(origins_t)

        # 世界坐标 → 体素坐标
        voxel_size = self.voxel_size
        grid_shape = self.grid_shape
        o = (origins_t - self.min_bound_t) / voxel_size  # 体素坐标系起点
        d = dirs_t  # 方向不变

        # 初始体素索引
        idx = torch.floor(o).long()  # [N, 3]
        step = torch.sign(d).long()  # [N, 3]

        # t_max: 到下一个体素边界的距离
        t_max = torch.full((N, 3), float('inf'), device=DEVICE)
        t_delta = torch.full((N, 3), float('inf'), device=DEVICE)

        for i in range(3):
            # 正方向：下一个整数边界
            # 负方向：当前整数边界
            next_boundary = torch.where(
                step[:, i] > 0,
                (idx[:, i].float() + 1) * voxel_size,
                idx[:, i].float() * voxel_size
            )
            # t_max = (boundary - origin) / direction
            safe_dir = torch.where(d[:, i].abs() > 1e-10, d[:, i], torch.full_like(d[:, i], 1e-10))
            t_max[:, i] = (next_boundary + self.min_bound[i] - origins_t[:, i]) / safe_dir

            # t_delta = voxel_size / |direction|
            t_delta[:, i] = voxel_size / safe_dir.abs()

        # 遍历
        t_hit = torch.full((N,), float(max_dist), device=DEVICE)
        max_steps = int(max_dist / voxel_size) * 3

        for _ in range(max_steps):
            # 检查当前体素是否在网格内且被占用
            in_bounds = (idx[:, 0] >= 0) & (idx[:, 0] < grid_shape[0]) & \
                        (idx[:, 1] >= 0) & (idx[:, 1] < grid_shape[1]) & \
                        (idx[:, 2] >= 0) & (idx[:, 2] < grid_shape[2])

            # 安全索引（避免越界）
            safe_idx = idx.clamp(
                torch.zeros(3, dtype=torch.long, device=DEVICE),
                torch.tensor(grid_shape, dtype=torch.long, device=DEVICE) - 1
            )
            occupied = self.grid_t[safe_idx[:, 0], safe_idx[:, 1], safe_idx[:, 2]].bool()
            hit = in_bounds & occupied

            # 记录命中距离（取 t_max 的最小值作为进入体素的距离）
            t_enter = t_max.min(dim=1).values
            t_hit = torch.where(hit & (t_enter < t_hit), t_enter, t_hit)

            # 步进到下一个体素（向量化）
            dim = t_max.argmin(dim=1)  # [N]
            # scatter 更新 idx 和 t_max
            idx_update = step[torch.arange(N, device=DEVICE), dim]
            t_max_update = t_delta[torch.arange(N, device=DEVICE), dim]
            idx[torch.arange(N, device=DEVICE), dim] += idx_update
            t_max[torch.arange(N, device=DEVICE), dim] += t_max_update

            # 所有光线都已命中或超出范围时提前退出
            if (t_hit < float(max_dist)).all():
                break

        return t_hit.cpu().numpy()


def filter_occluded_by_voxels(voxel_grid, pts_cam_np, c2w_np):
    """纯 CPU: 方向计算 + Open3D C++ ray-casting"""
    cam_origin = c2w_np[:3, 3]
    R_c2w = c2w_np[:3, :3]
    dists = np.linalg.norm(pts_cam_np[:, :3], axis=1)
    dirs_cam = pts_cam_np[:, :3] / dists[:, None]
    dirs_world = (R_c2w @ dirs_cam.T).astype(np.float32).T
    origins = np.tile(cam_origin.astype(np.float32), (len(pts_cam_np), 1))
    t_hits = voxel_grid.batch_raycast(origins, dirs_world, max_dist=DEPTH_MAX)
    return t_hits >= dists - voxel_grid.voxel_size


# ──────────────────────────────────────────────────────────────────────
# 处理单帧
# ──────────────────────────────────────────────────────────────────────

def process_single_frame(target_frame, all_indices, bbox_cache,
                         cam_id, cam_intrinsics_gpu, cam_extrinsics_gpu,
                         world_pts_all_gpu, H, W):
    """处理单帧（混合 GPU/CPU），返回 depth map [H, W] 或 None"""
    t0 = time.time()

    frame_start = max(0, target_frame - N_BEFORE)
    frame_end = min(all_indices[-1] + 1, target_frame + N_AFTER + 1)
    supp_frames = [f for f in range(frame_start, frame_end) if f != target_frame]

    # ── 构建体素网格（CPU 过滤 + CPU 体素）──
    voxel_frame_indices = supp_frames + [target_frame] if not VOXEL_ALL else all_indices
    voxel_world_list = []
    for f in voxel_frame_indices:
        for cid in CAM_LIST:
            if (f, cid) not in world_pts_all_gpu:
                continue
            world_pts_np = world_pts_all_gpu[(f, cid)].cpu().numpy()
            if f != target_frame or not MERGE_TARGET:
                dyn_mask = filter_dynamic_objects_cpu(world_pts_np, bbox_cache, f)
                world_pts_np = world_pts_np[dyn_mask]
            voxel_world_list.append(world_pts_np[:, :3])

    if not voxel_world_list:
        return None

    voxel_all_points = np.concatenate(voxel_world_list, axis=0)
    t1 = time.time()

    voxel_grid = VoxelOccupancyGrid(voxel_all_points, voxel_size=VOXEL_SIZE)
    t2 = time.time()

    if BBOX_VOXEL and target_frame in bbox_cache:
        centers, halves = bbox_cache[target_frame]
        bboxes_np = np.concatenate([centers.cpu().numpy(), halves.cpu().numpy() * 2.0], axis=1)
        voxel_grid.add_bboxes(bboxes_np)

    # ── 预计算 w2c（一次）──
    c2w_t = cam_extrinsics_gpu[target_frame][cam_id]
    w2c_t = torch.inverse(c2w_t)
    c2w_np = c2w_t.cpu().numpy()

    # ── 补充帧：CPU 过滤 → GPU 批量变换 → CPU 遮挡过滤 ──
    supp_world_list = []
    for f in supp_frames:
        if (f, cam_id) not in world_pts_all_gpu:
            continue
        world_pts_np = world_pts_all_gpu[(f, cam_id)].cpu().numpy()
        dyn_mask = filter_dynamic_objects_cpu(world_pts_np, bbox_cache, f)
        supp_world_list.append(world_pts_np[dyn_mask])

    if supp_world_list:
        supp_world_np = np.concatenate(supp_world_list, axis=0)
        # GPU 批量变换（一次 .cpu()）
        supp_world_t = torch.from_numpy(supp_world_np).float().to(DEVICE)
        supp_cam_t = world_to_camera_gpu(supp_world_t, w2c_t)
        supp_cam = supp_cam_t.cpu().numpy()
    else:
        supp_cam = np.empty((0, 4))
    t3 = time.time()

    # Open3D C++ ray-casting（CPU，BVH 加速）
    supp_visible = filter_occluded_by_voxels(voxel_grid, supp_cam, c2w_np)
    supp_filtered = supp_cam[supp_visible]
    t4 = time.time()

    # ── 合并目标帧 + 补充帧 ──
    if MERGE_TARGET:
        if (target_frame, cam_id) in world_pts_all_gpu:
            target_world_np = world_pts_all_gpu[(target_frame, cam_id)].cpu().numpy()
            target_world_t = torch.from_numpy(target_world_np).float().to(DEVICE)
            target_cam = world_to_camera_gpu(target_world_t, w2c_t).cpu().numpy()
        else:
            target_cam = np.empty((0, 4))
        vis_cam = np.concatenate([target_cam, supp_filtered], axis=0)
    else:
        vis_cam = supp_filtered

    if len(vis_cam) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # GPU 投影生成深度图
    K_t = cam_intrinsics_gpu[cam_id]['K']
    vis_cam_t = torch.from_numpy(vis_cam).float().to(DEVICE)
    depth_map = points_to_depth_map_gpu(vis_cam_t, K_t, W, H)
    t5 = time.time()

    print(f'    帧{target_frame:03d}_cam{cam_id}: 体素{t2-t1:.1f}s 补充帧{t3-t2:.1f}s '
          f'过滤{t4-t3:.1f}s 投影{t5-t4:.1f}s 合计{t5-t0:.1f}s')
    return depth_map


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────

def main():
    print(f'[INFO] 设备: {DEVICE}')

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

        # ── 预计算 bbox（GPU）──
        bbox_cache = precompute_bboxes_gpu(instances_info, frame_instances, all_indices)
        print(f'[INFO] 预计算 bbox: {len(bbox_cache)} 帧')

        # ── 预加载 intrinsics + K_inv（GPU）──
        cam_intrinsics_gpu = {}
        for cam_id in CAM_LIST:
            K = load_intrinsics(os.path.join(scene_dir, 'intrinsics', f'{cam_id}.txt'))
            K_t = torch.from_numpy(K).float().to(DEVICE)
            K_inv_t = torch.inverse(K_t)
            cam_intrinsics_gpu[cam_id] = {'K': K_t, 'K_inv': K_inv_t}

        # ── 预加载 extrinsics（GPU）──
        cam_extrinsics_gpu = {}
        for f_idx in all_indices:
            cam_extrinsics_gpu[f_idx] = {}
            for cam_id in CAM_LIST:
                ext_path = os.path.join(scene_dir, 'extrinsics', f'{f_idx:03d}_{cam_id}.txt')
                if os.path.exists(ext_path):
                    cam_extrinsics_gpu[f_idx][cam_id] = torch.from_numpy(load_txt_matrix(ext_path)).float().to(DEVICE)

        # ── 预加载所有 world points（GPU）──
        ref_dm = os.path.join(scene_dir, 'depth_map', f'{all_indices[0]:03d}_{RENDER_CAM_LIST[0]}.npz')
        if os.path.exists(ref_dm):
            H, W = np.load(ref_dm)['depth'].shape
        else:
            H, W = 900, 1600

        print(f'[INFO] 预加载 world points (GPU)...')
        world_pts_all_gpu = {}
        for f_idx in tqdm(all_indices, desc='预加载', leave=False):
            for cam_id in CAM_LIST:
                if f_idx not in cam_extrinsics_gpu or cam_id not in cam_extrinsics_gpu[f_idx]:
                    continue
                dm_path = os.path.join(scene_dir, 'depth_map', f'{f_idx:03d}_{cam_id}.npz')
                if not os.path.exists(dm_path):
                    continue
                depth = load_depth_map(dm_path)
                cam_pts = depth_to_camera_points_gpu(depth, cam_intrinsics_gpu[cam_id]['K_inv'])
                if len(cam_pts) == 0:
                    continue
                world_pts = camera_to_world_gpu(cam_pts, cam_extrinsics_gpu[f_idx][cam_id])
                world_pts_all_gpu[(f_idx, cam_id)] = world_pts
        print(f'[INFO] 预加载完成: {len(world_pts_all_gpu)} 个帧×相机组合')

        scene_start = time.time()
        n_done = 0
        n_skip = 0
        frame_indices = [f for f in all_indices if f >= START_FRAME]
        cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']

        for frame_idx in tqdm(frame_indices, desc=f"帧 {scene_name}", leave=False):
            for cam_id in RENDER_CAM_LIST:
                out_path = os.path.join(out_scene_dir, f'{frame_idx:03d}_{cam_id}.npz')
                if os.path.exists(out_path):
                    n_skip += 1
                    continue

                depth_map = process_single_frame(
                    frame_idx, all_indices, bbox_cache,
                    cam_id, cam_intrinsics_gpu, cam_extrinsics_gpu, world_pts_all_gpu, H, W
                )

                if depth_map is None:
                    continue

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
