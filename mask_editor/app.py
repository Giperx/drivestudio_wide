"""
DriveStudio Mask Editor - Web-based tool for correcting dynamic masks
Start with --dataset to select: nuscenes, lyft1224, lyft1920
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import json
import io
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label
from collections import defaultdict

app = Flask(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    'nuscenes': {
        'name': 'nuScenes',
        'data_root': os.path.join(_PROJECT_ROOT, 'data/nuscenes/processed_10Hz/trainval'),
        'save_root': os.path.join(_PROJECT_ROOT, 'data/nuscenes/corrected_masks/trainval'),
        'mask_dir': 'dynamic_masks',
        'cameras': [
            {'id': 0, 'name': 'CAM_FRONT'},
            {'id': 1, 'name': 'CAM_FRONT_LEFT'},
            {'id': 2, 'name': 'CAM_FRONT_RIGHT'},
            {'id': 3, 'name': 'CAM_BACK_LEFT'},
            {'id': 4, 'name': 'CAM_BACK_RIGHT'},
            {'id': 5, 'name': 'CAM_BACK'},
        ],
        'mask_types': ['human', 'vehicle', 'all'],
        'default_camera': 5,
        'default_mask_type': 'vehicle',
        'default_fps': 8,
    },
    'lyft1224': {
        'name': 'Lyft1224',
        'data_root': os.path.join(_PROJECT_ROOT, 'data/lyft/processed/lyft_train1224'),
        'save_root': os.path.join(_PROJECT_ROOT, 'data/lyft/processed/lyft_train1224_corrected'),
        'mask_dir': 'dynamic_masks',
        'cameras': [
            {'id': 0, 'name': 'CAM_FRONT'},
            {'id': 1, 'name': 'CAM_FRONT_LEFT'},
            {'id': 2, 'name': 'CAM_FRONT_RIGHT'},
            {'id': 3, 'name': 'CAM_BACK_LEFT'},
            {'id': 4, 'name': 'CAM_BACK_RIGHT'},
            {'id': 5, 'name': 'CAM_BACK'},
        ],
        'mask_types': ['human', 'vehicle', 'other_dynamics', 'all'],
        'default_camera': 5,
        'default_mask_type': 'vehicle',
        'default_fps': 5,
    },
    'lyft1920': {
        'name': 'Lyft1920',
        'data_root': os.path.join(_PROJECT_ROOT, 'data/lyft/processed/lyft_train1920'),
        'save_root': os.path.join(_PROJECT_ROOT, 'data/lyft/processed/lyft_train1920_corrected'),
        'mask_dir': 'dynamic_masks',
        'cameras': [
            {'id': 0, 'name': 'CAM_FRONT'},
            {'id': 1, 'name': 'CAM_FRONT_LEFT'},
            {'id': 2, 'name': 'CAM_FRONT_RIGHT'},
            {'id': 3, 'name': 'CAM_BACK_LEFT'},
            {'id': 4, 'name': 'CAM_BACK_RIGHT'},
            {'id': 5, 'name': 'CAM_BACK'},
        ],
        'mask_types': ['human', 'vehicle', 'other_dynamics', 'all'],
        'default_camera': 5,
        'default_mask_type': 'vehicle',
        'default_fps': 5,
    },
}

# Active dataset config (set at startup)
DS = DATASETS['nuscenes']
DATA_ROOT = DS['data_root']
SAVE_ROOT = DS['save_root']
MASK_DIR = DS['mask_dir']


def get_mask_path(scene, mask_type, frame):
    corrected = os.path.join(SAVE_ROOT, scene, MASK_DIR, mask_type, f'{frame}.png')
    if os.path.exists(corrected):
        return corrected
    return os.path.join(DATA_ROOT, scene, MASK_DIR, mask_type, f'{frame}.png')


def load_mask_array(scene, mask_type, frame):
    path = get_mask_path(scene, mask_type, frame)
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('L')
    return np.array(img)


def ensure_mask_in_corrected(scene, mask_type, frame):
    """Ensure a mask exists in the corrected directory. Copy from original if not."""
    corrected = os.path.join(SAVE_ROOT, scene, MASK_DIR, mask_type, f'{frame}.png')
    if os.path.exists(corrected):
        return corrected
    original = os.path.join(DATA_ROOT, scene, MASK_DIR, mask_type, f'{frame}.png')
    if os.path.exists(original):
        os.makedirs(os.path.dirname(corrected), exist_ok=True)
        shutil.copy2(original, corrected)
        return corrected
    return None


def get_image_size(scene, frame):
    """Get image dimensions from actual file, fallback to defaults."""
    img_path = os.path.join(DATA_ROOT, scene, 'images', f'{frame}.jpg')
    if os.path.exists(img_path):
        img = Image.open(img_path)
        return img.size  # (width, height)
    return (1600, 900)


# ---------- Rectangle Decomposition ----------

def decompose_component(component, min_area=20):
    ys, xs = np.where(component)
    if len(xs) < min_area:
        return []

    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    comp_bbox = component[y1:y2+1, x1:x2+1].astype(bool)
    bw, bh = x2 - x1 + 1, y2 - y1 + 1

    if comp_bbox.sum() == bw * bh:
        return [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'area': int(comp_bbox.sum())}]

    x_set = {0, bw}
    y_set = {0, bh}
    for r in range(bh):
        row = comp_bbox[r]
        if not row.any():
            continue
        diffs = np.diff(row.astype(int))
        for c in np.where(diffs == 1)[0]:
            x_set.add(int(c + 1))
        for c in np.where(diffs == -1)[0]:
            x_set.add(int(c + 1))
        filled = np.where(row)[0]
        if len(filled) > 0:
            x_set.add(int(filled[0]))
            x_set.add(int(filled[-1] + 1))

    for c in range(bw):
        col = comp_bbox[:, c]
        if not col.any():
            continue
        diffs = np.diff(col.astype(int))
        for r in np.where(diffs == 1)[0]:
            y_set.add(int(r + 1))
        for r in np.where(diffs == -1)[0]:
            y_set.add(int(r + 1))
        filled = np.where(col)[0]
        if len(filled) > 0:
            y_set.add(int(filled[0]))
            y_set.add(int(filled[-1] + 1))

    x_coords = sorted(x_set)
    y_coords = sorted(y_set)
    nx, ny = len(x_coords) - 1, len(y_coords) - 1

    if nx == 0 or ny == 0:
        return [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'area': int(comp_bbox.sum())}]

    grid = np.zeros((ny, nx), dtype=bool)
    for r in range(ny):
        yr = y_coords[r]
        for c in range(nx):
            xc = x_coords[c]
            grid[r][c] = comp_bbox[yr, xc]

    rectangles = []
    max_iter = 200

    for _ in range(max_iter):
        best = None
        best_score = (0, 0, 0)

        for c in range(nx):
            heights = np.zeros(ny, dtype=int)
            for r in range(ny - 1, -1, -1):
                if grid[r][c]:
                    heights[r] = (heights[r + 1] + 1) if r + 1 < ny else 1
                else:
                    heights[r] = 0

            for r in range(ny):
                if not grid[r][c]:
                    continue
                min_h = heights[r]
                for c2 in range(c, nx):
                    if not grid[r][c2] or heights[r] == 0:
                        break
                    h_at_c2 = 0
                    for rr in range(r, ny):
                        if grid[rr][c2]:
                            h_at_c2 += 1
                        else:
                            break
                    min_h = min(min_h, h_at_c2)
                    if min_h == 0:
                        break
                    pixel_w = x_coords[c2 + 1] - x_coords[c]
                    pixel_h = y_coords[r + min_h] - y_coords[r]
                    pixel_area = pixel_w * pixel_h
                    score = (pixel_area, pixel_h, pixel_w)
                    if score > best_score:
                        best_score = score
                        best = (r, c, r + min_h - 1, c2)

        if best is None:
            break

        r1, c1, r2, c2 = best
        px_x1 = x1 + x_coords[c1]
        px_y1 = y1 + y_coords[r1]
        px_x2 = x1 + x_coords[c2 + 1] - 1
        px_y2 = y1 + y_coords[r2 + 1] - 1

        pixel_area = (px_x2 - px_x1 + 1) * (px_y2 - px_y1 + 1)
        if pixel_area < min_area:
            break

        rectangles.append({
            'x1': px_x1, 'y1': px_y1,
            'x2': px_x2, 'y2': px_y2,
            'area': pixel_area,
        })

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                grid[r][c] = False

    return rectangles


# ---------- API Routes ----------

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/config')
def api_config():
    """Return current dataset configuration for the frontend."""
    return jsonify({
        'cameras': DS['cameras'],
        'mask_types': DS['mask_types'],
        'default_camera': DS['default_camera'],
        'default_mask_type': DS['default_mask_type'],
        'default_fps': DS.get('default_fps', 8),
        'dataset_name': DS['name'],
    })


@app.route('/api/scenes')
def api_scenes():
    if not os.path.exists(DATA_ROOT):
        return jsonify([])
    scenes = sorted([d for d in os.listdir(DATA_ROOT)
                     if os.path.isdir(os.path.join(DATA_ROOT, d))])
    return jsonify(scenes)


@app.route('/api/scene/<scene_id>/frames')
def api_frames(scene_id):
    img_dir = os.path.join(DATA_ROOT, scene_id, 'images')
    if not os.path.exists(img_dir):
        return jsonify({'timesteps': [], 'cameras': DS['cameras']})

    files = sorted(os.listdir(img_dir))
    ts_set = sorted(set(f.split('_')[0] for f in files if f.endswith('.jpg')))
    return jsonify({'timesteps': ts_set, 'cameras': DS['cameras']})


@app.route('/api/image/<scene_id>/<frame>')
def api_image(scene_id, frame):
    return send_from_directory(os.path.join(DATA_ROOT, scene_id, 'images'), f'{frame}.jpg')


@app.route('/api/mask/<scene_id>/<mask_type>/<frame>')
def api_mask(scene_id, mask_type, frame):
    path = get_mask_path(scene_id, mask_type, frame)
    if not os.path.exists(path):
        w, h = get_image_size(scene_id, frame)
        blank = Image.new('L', (w, h), 0)
        buf = io.BytesIO()
        blank.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return send_file(path, mimetype='image/png')


@app.route('/api/mask_contours/<scene_id>/<mask_type>/<frame>')
def api_mask_contours(scene_id, mask_type, frame):
    arr = load_mask_array(scene_id, mask_type, frame)
    if arr is None:
        w, h = get_image_size(scene_id, frame)
        return jsonify({'regions': [], 'width': w, 'height': h})

    h, w = arr.shape
    binary = (arr > 128).astype(np.uint8)

    labeled, num_features = scipy_label(binary)

    regions = []
    region_id = 0

    for i in range(1, num_features + 1):
        component = (labeled == i)
        rects = decompose_component(component, min_area=20)

        for rect in rects:
            region_id += 1
            cx = (rect['x1'] + rect['x2']) // 2
            cy = (rect['y1'] + rect['y2']) // 2
            regions.append({
                'id': region_id,
                'bbox': [rect['x1'], rect['y1'], rect['x2'], rect['y2']],
                'area': rect['area'],
                'centroid': [cx, cy],
            })

    return jsonify({'regions': regions, 'width': w, 'height': h})


@app.route('/api/save_mask', methods=['POST'])
def api_save_mask():
    data = request.json
    scene = data['scene']
    mask_type = data['mask_type']
    frame = data['frame']
    additions = data.get('additions', [])
    removals = data.get('removals', [])

    arr = load_mask_array(scene, mask_type, frame)
    if arr is None:
        w, h = get_image_size(scene, frame)
        arr = np.zeros((h, w), dtype=np.uint8)

    for rect in additions:
        x1, y1, x2, y2 = int(rect['x1']), int(rect['y1']), int(rect['x2']), int(rect['y2'])
        x1 = max(0, min(arr.shape[1] - 1, x1))
        x2 = max(0, min(arr.shape[1] - 1, x2))
        y1 = max(0, min(arr.shape[0] - 1, y1))
        y2 = max(0, min(arr.shape[0] - 1, y2))
        arr[y1:y2 + 1, x1:x2 + 1] = 255

    for rect in removals:
        x1, y1, x2, y2 = int(rect['x1']), int(rect['y1']), int(rect['x2']), int(rect['y2'])
        x1 = max(0, min(arr.shape[1] - 1, x1))
        x2 = max(0, min(arr.shape[1] - 1, x2))
        y1 = max(0, min(arr.shape[0] - 1, y1))
        y2 = max(0, min(arr.shape[0] - 1, y2))
        arr[y1:y2 + 1, x1:x2 + 1] = 0

    save_dir = os.path.join(SAVE_ROOT, scene, MASK_DIR, mask_type)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{frame}.png')
    Image.fromarray(arr, mode='L').save(save_path)

    return jsonify({'success': True, 'path': save_path,
                    'num_additions': len(additions), 'num_removals': len(removals)})


@app.route('/api/save_all_masks', methods=['POST'])
def api_save_all_masks():
    data = request.json
    scene = data['scene']
    frame = data['frame']

    # Determine which mask types to merge into "all"
    merge_types = [mt for mt in DS['mask_types'] if mt != 'all']

    if not merge_types:
        # No individual types, just ensure "all" exists
        result = ensure_mask_in_corrected(scene, 'all', frame)
        if result:
            return jsonify({'success': True, 'path': result})
        return jsonify({'success': False, 'error': 'No all mask found'})

    # Ensure all individual masks exist in corrected dir
    for mt in merge_types:
        ensure_mask_in_corrected(scene, mt, frame)

    # Load and merge
    combined = None
    for mt in merge_types:
        arr = load_mask_array(scene, mt, frame)
        if arr is not None:
            if combined is None:
                combined = arr.copy()
            else:
                combined = np.maximum(combined, arr)

    if combined is None:
        return jsonify({'success': False, 'error': f'No masks found for types: {merge_types}'})

    save_dir = os.path.join(SAVE_ROOT, scene, MASK_DIR, 'all')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{frame}.png')
    Image.fromarray(combined, mode='L').save(save_path)

    return jsonify({'success': True, 'path': save_path})


@app.route('/api/save_full_mask', methods=['POST'])
def api_save_full_mask():
    """Save a full-white mask for the given scene/frame/mask_type (entire image is mask)."""
    data = request.json
    scene = data['scene']
    mask_type = data['mask_type']
    frame = data['frame']

    w, h = get_image_size(scene, frame)
    arr = np.full((h, w), 255, dtype=np.uint8)

    save_dir = os.path.join(SAVE_ROOT, scene, MASK_DIR, mask_type)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{frame}.png')
    Image.fromarray(arr, mode='L').save(save_path)

    return jsonify({'success': True, 'path': save_path, 'width': w, 'height': h})


@app.route('/api/check_saved/<scene_id>/<mask_type>/<frame>')
def api_check_saved(scene_id, mask_type, frame):
    corrected = os.path.join(SAVE_ROOT, scene_id, MASK_DIR, mask_type, f'{frame}.png')
    return jsonify({'exists': os.path.exists(corrected), 'path': corrected})


# ---------- Main ----------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mask Editor')
    parser.add_argument('--dataset', default='nuscenes', choices=DATASETS.keys(), help='Dataset to use')
    parser.add_argument('--port', type=int, default=5050, help='Port number')
    parser.add_argument('--host', default='0.0.0.0', help='Host')
    args = parser.parse_args()

    DS = DATASETS[args.dataset]
    DATA_ROOT = DS['data_root']
    SAVE_ROOT = DS['save_root']
    MASK_DIR = DS['mask_dir']

    os.makedirs(SAVE_ROOT, exist_ok=True)
    print(f"Dataset:    {DS['name']} ({args.dataset})")
    print(f"Data root:  {DATA_ROOT}")
    print(f"Save root:  {SAVE_ROOT}")
    print(f"Mask dir:   {MASK_DIR}")
    print(f"Cameras:    {[c['id'] for c in DS['cameras']]}")
    print(f"Mask types: {DS['mask_types']}")
    print(f"Server:     http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
