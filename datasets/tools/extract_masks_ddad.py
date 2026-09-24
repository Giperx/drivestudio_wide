"""
@file   extract_masks_ddad.py
@brief  Simplified extract_masks for DDAD dataset — sky masks only.

Using SegFormer (Cityscapes, 19 classes). Sky class index = 10.

Usage:
    conda activate segformer
    CUDA_VISIBLE_DEVICES=0 python datasets/tools/extract_masks_ddad.py \
        --data_root data/ddad_process/valid \
        --segformer_path=./SegFormer \
        --checkpoint=./SegFormer/segformer.b5.1024x1024.city.160k.pth \
        --start_idx 0 \
        --num_scenes 50
"""

import os
import numpy as np
import imageio
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor


if __name__ == "__main__":
    parser = ArgumentParser()
    # Data configs
    parser.add_argument('--data_root', type=str, default='data/ddad_process/valid')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--num_scenes', type=int, default=53)
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")

    # SegFormer configs
    parser.add_argument('--segformer_path', type=str, default='./SegFormer')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(
            args.segformer_path, 'local_configs', 'segformer', 'B5',
            'segformer.b5.1024x1024.city.160k.py')
    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            args.segformer_path, 'pretrained',
            'segformer.b5.1024x1024.city.160k.pth')

    scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    for scene_i, scene_id in enumerate(tqdm(scene_ids_list, 'Extracting sky masks')):
        scene_id = str(scene_id).zfill(3)
        img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname)

        if not os.path.isdir(img_dir):
            print(f"[WARN] scene {scene_id}: images dir not found, skipping")
            continue

        # Create sky_masks dir
        sky_mask_dir = os.path.join(args.data_root, scene_id, "sky_masks")
        os.makedirs(sky_mask_dir, exist_ok=True)

        flist = sorted(glob(os.path.join(img_dir, '*')))
        for fpath in tqdm(flist, f'scene[{scene_id}]', leave=False):
            fbase = os.path.splitext(os.path.basename(fpath))[0]

            if args.ignore_existing and os.path.exists(
                    os.path.join(sky_mask_dir, f"{fbase}.png")):
                continue

            # SegFormer inference
            result = inference_segmentor(model, fpath)
            mask = result[0].astype(np.uint8)

            # Sky class = 10 in Cityscapes
            sky_mask = np.isin(mask, [10])
            imageio.imwrite(
                os.path.join(sky_mask_dir, f"{fbase}.png"),
                sky_mask.astype(np.uint8) * 255)

    print("Done.")
