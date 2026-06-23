"""
DriveStudio nuScenes Mask Editor - Web-based tool for correcting dynamic masks
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
DATA_ROOT = os.environ.get('DATA_ROOT', os.path.join(_PROJECT_ROOT, 'data/nuscenes/processed_10Hz/trainval'))
SAVE_ROOT = os.environ.get('SAVE_ROOT', os.path.join(_PROJECT_ROOT, 'data/nuscenes/corrected_masks/trainval'))

CAMERAS = [
    {'id': 0, 'name': 'CAM_FRONT'},
    {'id': 1, 'name': 'CAM_FRONT_LEFT'},
    {'id': 2, 'name': 'CAM_FRONT_RIGHT'},
    {'id': 3, 'name': 'CAM_BACK_LEFT'},
    {'id': 4, 'name': 'CAM_BACK_RIGHT'},
    {'id': 5, 'name': 'CAM_BACK'},
]


def get_mask_path(scene, mask_type, frame):
    corrected = os.path.join(SAVE_ROOT, scene, 'dynamic_masks', mask_type, f'{frame}.png')
    if os.path.exists(corrected):
        return corrected
    return os.path.join(DATA_ROOT, scene, 'dynamic_masks', mask_type, f'{frame}.png')


def load_mask_array(scene, mask_type, frame):
    path = get_mask_path(scene, mask_type, frame)
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('L')
    return np.array(img)


def ensure_mask_in_corrected(scene, mask_type, frame):
    """Ensure a mask exists in the corrected directory. Copy from original if not."""
    corrected = os.path.join(SAVE_ROOT, scene, 'dynamic_masks', mask_type, f'{frame}.png')
    if os.path.exists(corrected):
        return corrected
    original = os.path.join(DATA_ROOT, scene, 'dynamic_masks', mask_type, f'{frame}.png')
    if os.path.exists(original):
        os.makedirs(os.path.dirname(corrected), exist_ok=True)
        shutil.copy2(original, corrected)
        return corrected
    return None


# ---------- Rectangle Decomposition ----------

def decompose_component(component, min_area=20):
    """
    Decompose a connected component (union of axis-aligned rectangles)
    back into individual rectangles using grid-based greedy decomposition.
    """
    ys, xs = np.where(component)
    if len(xs) < min_area:
        return []

    # Get bounding box
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    comp_bbox = component[y1:y2+1, x1:x2+1].astype(bool)
    bw, bh = x2 - x1 + 1, y2 - y1 + 1

    # If already a rectangle, return it directly
    if comp_bbox.sum() == bw * bh:
        return [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                 'area': int(comp_bbox.sum())}]

    # Build grid from transition points within the bbox region
    x_set = {0, bw}
    y_set = {0, bh}
    for r in range(bh):
        row = comp_bbox[r]
        if not row.any():
            continue
        # Find transitions: False->True and True->False
        diffs = np.diff(row.astype(int))
        for c in np.where(diffs == 1)[0]:    # False -> True
            x_set.add(int(c + 1))
        for c in np.where(diffs == -1)[0]:   # True -> False
            x_set.add(int(c + 1))
        # Also add first/last filled in row
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
        return [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                 'area': int(comp_bbox.sum())}]

    # Build boolean grid: grid[r][c] = True if cell is in mask
    grid = np.zeros((ny, nx), dtype=bool)
    for r in range(ny):
        yr = y_coords[r]
        for c in range(nx):
            xc = x_coords[c]
            grid[r][c] = comp_bbox[yr, xc]

    # Greedy decomposition: vertical-first, prefer tallest and largest rectangles
    rectangles = []
    max_iter = 200  # safety limit

    for _ in range(max_iter):
        best = None
        best_score = (0, 0, 0)  # (area, height, width) for tie-breaking

        for c in range(nx):
            # For each column, compute consecutive height going downward
            heights = np.zeros(ny, dtype=int)
            for r in range(ny - 1, -1, -1):
                if grid[r][c]:
                    heights[r] = (heights[r + 1] + 1) if r + 1 < ny else 1
                else:
                    heights[r] = 0

            for r in range(ny):
                if not grid[r][c]:
                    continue
                # Expand rightward, tracking minimum height
                min_h = heights[r]
                for c2 in range(c, nx):
                    if not grid[r][c2] or heights[r] == 0:
                        break
                    # Recalc min height for this column
                    h_at_c2 = 0
                    for rr in range(r, ny):
                        if grid[rr][c2]:
                            h_at_c2 += 1
                        else:
                            break
                    min_h = min(min_h, h_at_c2)
                    if min_h == 0:
                        break
                    # Score: prioritize area, then height (vertical-first)
                    pixel_w = x_coords[c2 + 1] - x_coords[c]
                    pixel_h = y_coords[r + min_h] - y_coords[r]
                    pixel_area = pixel_w * pixel_h
                    score = (pixel_area, pixel_h, pixel_w)  # tie-break: taller wins
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

        # Check pixel area against min_area
        pixel_area = (px_x2 - px_x1 + 1) * (px_y2 - px_y1 + 1)
        if pixel_area < min_area:
            break

        rectangles.append({
            'x1': px_x1, 'y1': px_y1,
            'x2': px_x2, 'y2': px_y2,
            'area': pixel_area,
        })

        # Remove this rectangle from the grid
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                grid[r][c] = False

    return rectangles


# ---------- API Routes ----------

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


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
        return jsonify({'timesteps': [], 'cameras': CAMERAS})

    files = sorted(os.listdir(img_dir))
    ts_set = sorted(set(f.split('_')[0] for f in files if f.endswith('.jpg')))
    return jsonify({'timesteps': ts_set, 'cameras': CAMERAS})


@app.route('/api/image/<scene_id>/<frame>')
def api_image(scene_id, frame):
    return send_from_directory(os.path.join(DATA_ROOT, scene_id, 'images'), f'{frame}.jpg')


@app.route('/api/mask/<scene_id>/<mask_type>/<frame>')
def api_mask(scene_id, mask_type, frame):
    path = get_mask_path(scene_id, mask_type, frame)
    if not os.path.exists(path):
        img = Image.new('L', (1600, 900), 0)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return send_file(path, mimetype='image/png')


@app.route('/api/mask_contours/<scene_id>/<mask_type>/<frame>')
def api_mask_contours(scene_id, mask_type, frame):
    """Decompose mask into individual rectangles (handles overlapping)."""
    arr = load_mask_array(scene_id, mask_type, frame)
    if arr is None:
        return jsonify({'regions': [], 'width': 1600, 'height': 900})

    h, w = arr.shape
    binary = (arr > 128).astype(np.uint8)

    # Connected component labeling
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

    # Load existing mask
    arr = load_mask_array(scene, mask_type, frame)
    if arr is None:
        arr = np.zeros((900, 1600), dtype=np.uint8)

    # Apply additions (draw rectangles as white)
    for rect in additions:
        x1, y1, x2, y2 = int(rect['x1']), int(rect['y1']), int(rect['x2']), int(rect['y2'])
        x1 = max(0, min(arr.shape[1] - 1, x1))
        x2 = max(0, min(arr.shape[1] - 1, x2))
        y1 = max(0, min(arr.shape[0] - 1, y1))
        y2 = max(0, min(arr.shape[0] - 1, y2))
        arr[y1:y2 + 1, x1:x2 + 1] = 255

    # Apply removals (clear rectangle regions)
    for rect in removals:
        x1, y1, x2, y2 = int(rect['x1']), int(rect['y1']), int(rect['x2']), int(rect['y2'])
        x1 = max(0, min(arr.shape[1] - 1, x1))
        x2 = max(0, min(arr.shape[1] - 1, x2))
        y1 = max(0, min(arr.shape[0] - 1, y1))
        y2 = max(0, min(arr.shape[0] - 1, y2))
        arr[y1:y2 + 1, x1:x2 + 1] = 0

    # Save to corrected directory
    save_dir = os.path.join(SAVE_ROOT, scene, 'dynamic_masks', mask_type)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{frame}.png')

    Image.fromarray(arr, mode='L').save(save_path)

    return jsonify({'success': True, 'path': save_path,
                    'num_additions': len(additions), 'num_removals': len(removals)})


@app.route('/api/save_all_masks', methods=['POST'])
def api_save_all_masks():
    """Regenerate 'all' mask from human + vehicle masks for a given frame.
    Also ensures human and vehicle masks are saved to corrected directory."""
    data = request.json
    scene = data['scene']
    frame = data['frame']

    # Ensure human and vehicle masks exist in corrected directory (copy from original if needed)
    ensure_mask_in_corrected(scene, 'human', frame)
    ensure_mask_in_corrected(scene, 'vehicle', frame)

    # Load human and vehicle masks
    human = load_mask_array(scene, 'human', frame)
    vehicle = load_mask_array(scene, 'vehicle', frame)

    if human is None and vehicle is None:
        return jsonify({'success': False, 'error': 'No human or vehicle masks found'})

    h = human.shape[0] if human is not None else (vehicle.shape[0] if vehicle is not None else 900)
    w = human.shape[1] if human is not None else (vehicle.shape[1] if vehicle is not None else 1600)

    combined = np.zeros((h, w), dtype=np.uint8)
    if human is not None:
        combined = np.maximum(combined, human)
    if vehicle is not None:
        combined = np.maximum(combined, vehicle)

    save_dir = os.path.join(SAVE_ROOT, scene, 'dynamic_masks', 'all')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{frame}.png')
    Image.fromarray(combined, mode='L').save(save_path)

    return jsonify({'success': True, 'path': save_path})


@app.route('/api/save_all_masks_batch', methods=['POST'])
def api_save_all_masks_batch():
    """Regenerate human, vehicle, and 'all' masks for ALL frames of a scene.
    Copies unmodified masks from original to corrected directory."""
    data = request.json
    scene = data['scene']

    # Find all frames that have either human or vehicle masks
    human_dir = os.path.join(SAVE_ROOT, scene, 'dynamic_masks', 'human')
    vehicle_dir = os.path.join(SAVE_ROOT, scene, 'dynamic_masks', 'vehicle')
    human_orig = os.path.join(DATA_ROOT, scene, 'dynamic_masks', 'human')
    vehicle_orig = os.path.join(DATA_ROOT, scene, 'dynamic_masks', 'vehicle')

    frames = set()
    for d in [human_dir, vehicle_dir, human_orig, vehicle_orig]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.png'):
                    frames.add(f.replace('.png', ''))

    frames = sorted(frames)
    if not frames:
        return jsonify({'success': False, 'error': 'No frames found', 'count': 0})

    all_dir = os.path.join(SAVE_ROOT, scene, 'dynamic_masks', 'all')
    os.makedirs(all_dir, exist_ok=True)

    count = 0
    for frame in frames:
        # Ensure human and vehicle masks exist in corrected directory
        ensure_mask_in_corrected(scene, 'human', frame)
        ensure_mask_in_corrected(scene, 'vehicle', frame)

        # Load and combine
        human = load_mask_array(scene, 'human', frame)
        vehicle = load_mask_array(scene, 'vehicle', frame)
        if human is None and vehicle is None:
            continue

        h = human.shape[0] if human is not None else (vehicle.shape[0] if vehicle is not None else 900)
        w = human.shape[1] if human is not None else (vehicle.shape[1] if vehicle is not None else 1600)

        combined = np.zeros((h, w), dtype=np.uint8)
        if human is not None:
            combined = np.maximum(combined, human)
        if vehicle is not None:
            combined = np.maximum(combined, vehicle)

        save_path = os.path.join(all_dir, f'{frame}.png')
        Image.fromarray(combined, mode='L').save(save_path)
        count += 1

    return jsonify({'success': True, 'count': count, 'total_frames': len(frames)})


@app.route('/api/check_saved/<scene_id>/<mask_type>/<frame>')
def api_check_saved(scene_id, mask_type, frame):
    """Check if a corrected mask exists for this frame."""
    corrected = os.path.join(SAVE_ROOT, scene_id, 'dynamic_masks', mask_type, f'{frame}.png')
    return jsonify({'exists': os.path.exists(corrected), 'path': corrected})


# ---------- Main ----------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mask Editor')
    parser.add_argument('--data_root', default=DATA_ROOT, help='Data root directory')
    parser.add_argument('--save_root', default=SAVE_ROOT, help='Save root directory')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--host', default='0.0.0.0', help='Host')
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    SAVE_ROOT = args.save_root

    os.makedirs(SAVE_ROOT, exist_ok=True)
    print(f"Data root:  {DATA_ROOT}")
    print(f"Save root:  {SAVE_ROOT}")
    print(f"Server:     http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
