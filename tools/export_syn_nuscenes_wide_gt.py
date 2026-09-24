"""Resize syn-nuScenes camera 2 into full-pixel wide GT folders.

Camera 2 is CAM_BACK_WIDE_RECT_GT (5760x1080). The exported name keeps the
project-wide ``{frame}_5_wide.png`` pattern used by nuScenes and Lyft wide
views. No mask is written: the CARLA GT covers every pixel.

Example:
    python tools/export_syn_nuscenes_wide_gt.py \
        --processed_dir data/syn_nuscenes/processed_10Hz/trainval \
        --output_root data/syn_nuscenes
"""

import argparse
import os
from typing import Iterable, List, Sequence, Tuple

from PIL import Image

# Folder name is width x height, matching sparseWideFOVImages3_* layouts.
DEFAULT_SIZES = ((1554, 294), (1344, 252))
SOURCE_CAM_ID = 2
# Output slot stays 5 so loaders can share the existing ``*_5_wide.png`` names.
OUTPUT_CAM_TAG = 5


def parse_sizes(values: Sequence[str]) -> List[Tuple[int, int]]:
    sizes = []
    for value in values:
        width_str, height_str = value.lower().split("x")
        width, height = int(width_str), int(height_str)
        if width <= 0 or height <= 0:
            raise argparse.ArgumentTypeError(f"invalid size {value}")
        sizes.append((width, height))
    return sizes


def scene_ids(processed_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(processed_dir)):
        image_dir = os.path.join(processed_dir, name, "images")
        if os.path.isdir(image_dir):
            yield name


def source_frames(scene_dir: str) -> List[Tuple[str, str]]:
    image_dir = os.path.join(scene_dir, "images")
    frames = []
    suffix = f"_{SOURCE_CAM_ID}.jpg"
    for name in sorted(os.listdir(image_dir)):
        if not name.endswith(suffix):
            continue
        frame_idx = name[: -len(suffix)]
        if not frame_idx.isdigit():
            continue
        frames.append((frame_idx, os.path.join(image_dir, name)))
    return frames


def export_scene(processed_dir: str, scene_id: str, output_root: str, sizes: Sequence[Tuple[int, int]]) -> int:
    frames = source_frames(os.path.join(processed_dir, scene_id))
    if not frames:
        raise FileNotFoundError(
            f"no camera {SOURCE_CAM_ID} images under {processed_dir}/{scene_id}/images"
        )

    out_dirs = []
    for width, height in sizes:
        rgb_dir = os.path.join(
            output_root,
            f"syn_nuscenes_wide_gt_{width}x{height}",
            scene_id,
            "rgb",
        )
        os.makedirs(rgb_dir, exist_ok=True)
        out_dirs.append((width, height, rgb_dir))

    for frame_idx, source_path in frames:
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            for width, height, rgb_dir in out_dirs:
                resized = image.resize((width, height), Image.Resampling.LANCZOS)
                target = os.path.join(rgb_dir, f"{frame_idx}_{OUTPUT_CAM_TAG}_wide.png")
                resized.save(target)
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed_dir",
        default="data/syn_nuscenes/processed_10Hz/trainval",
        help="processed syn-nuScenes split, with one folder per scene id",
    )
    parser.add_argument(
        "--output_root",
        default="data/syn_nuscenes",
        help="parent of syn_nuscenes_wide_gt_{W}x{H}",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=[f"{width}x{height}" for width, height in DEFAULT_SIZES],
        help="output sizes as WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--scene_ids",
        nargs="*",
        default=None,
        help="scene ids to export; default is every scene under processed_dir",
    )
    args = parser.parse_args()
    sizes = parse_sizes(args.sizes)
    selected = args.scene_ids if args.scene_ids else list(scene_ids(args.processed_dir))
    if not selected:
        raise FileNotFoundError(f"no processed scenes under {args.processed_dir}")

    for scene_id in selected:
        count = export_scene(args.processed_dir, scene_id, args.output_root, sizes)
        print(f"scene {scene_id}: {count} frames -> " + ", ".join(f"{w}x{h}" for w, h in sizes))


if __name__ == "__main__":
    main()
