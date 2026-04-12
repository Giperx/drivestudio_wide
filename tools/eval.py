from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import numpy as np
import wandb
import logging
import argparse

import torch
from datasets.base.pixel_source import get_rays
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views
)

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


@torch.no_grad()
def save_single_wide_rgbs(
    trainer: BasicTrainer,
    dataset: DrivingDataset,
    output_dir: str,
    width_scale: float = 0.5,
):
    """Render and save per-frame per-camera RGBs with eval-only widened intrinsics.

    Output filename format: {frame_idx:03d}_{camera_idx}.png
    where camera_idx is the local camera index in the selected camera list.
    """
    os.makedirs(output_dir, exist_ok=True)
    trainer.set_eval()

    split = dataset.full_image_set
    camera_downscale = trainer._get_downscale_factor()
    num_items = len(split)

    for i in range(num_items):
        image_infos, cam_infos = split.get_image(i, camera_downscale)

        # Eval-only intrinsics tweak for pseudoGTWide generation.
        # 1) 修改内参和目标宽度
        H = int(cam_infos["height"].item())
        W = int(cam_infos["width"].item())
        new_intrinsics = cam_infos["intrinsics"].clone()
        # new_intrinsics[..., 0, 0] *= 0.5
        new_intrinsics[..., 0, 2] += W / 2
        # new_intrinsics[..., 0, :3] = new_intrinsics[..., 0, :3] * 2.0
        # new_intrinsics[..., 1, :3] = new_intrinsics[..., 1, :3] * 2.0
        cam_infos["intrinsics"] = new_intrinsics
        
        target_w = W * 2
        cam_infos["width"] = torch.tensor(target_w, dtype=cam_infos["width"].dtype)
        target_h = H
        # target_h = H * 2
        # cam_infos["height"] = torch.tensor(target_h, dtype=cam_infos["height"].dtype)

        # 2) 用新内参重建 rays（保持和 opacity 一致的 H x target_w）
        x, y = torch.meshgrid(
            torch.arange(target_w, device=new_intrinsics.device),
            torch.arange(target_h, device=new_intrinsics.device),
            indexing="xy",
        )
        origins, viewdirs, direction_norm = get_rays(
            x.flatten(), y.flatten(),
            cam_infos["camera_to_world"],
            cam_infos["intrinsics"],
        )
        image_infos["origins"] = origins.reshape(target_h, target_w, 3)
        image_infos["viewdirs"] = viewdirs.reshape(target_h, target_w, 3)
        image_infos["direction_norm"] = direction_norm.reshape(target_h, target_w, 1)
        image_infos["pixel_coords"] = torch.stack(
            [y / target_h, x / target_w], dim=-1
        ).float().reshape(target_h, target_w, 2)

        # 3) 同步索引类张量形状
        if "img_idx" in image_infos:
            img_id = image_infos["img_idx"].flatten()[0]
            image_infos["img_idx"] = torch.full(
                (target_h, target_w), img_id, dtype=image_infos["img_idx"].dtype, device=image_infos["img_idx"].device
            )
        if "frame_idx" in image_infos:
            frame_id = image_infos["frame_idx"].flatten()[0]
            image_infos["frame_idx"] = torch.full(
                (target_h, target_w), frame_id, dtype=image_infos["frame_idx"].dtype, device=image_infos["frame_idx"].device
            )
        if "normed_time" in image_infos:
            t = image_infos["normed_time"].flatten()[0]
            image_infos["normed_time"] = torch.full(
                (target_h, target_w), t, dtype=image_infos["normed_time"].dtype, device=image_infos["normed_time"].device
            )
                                 
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)

        outputs = trainer(image_infos=image_infos, camera_infos=cam_infos)
        rgb = outputs["rgb"].clamp(0.0, 1.0).cpu().numpy()
        rgb_uint8 = (rgb * 255.0).astype(np.uint8)

        cam_local_idx, frame_idx = split.datasource.parse_img_idx(split.split_indices[i])
        save_name = f"{frame_idx:03d}_{cam_local_idx}.png"
        save_path = os.path.join(output_dir, save_name)
        from imageio import v2 as imageio
        imageio.imwrite(save_path, rgb_uint8)

    logger.info(f"Saved single_wide RGB images to {output_dir}")

@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True,
    output_root: Optional[str] = None,
):
    trainer.set_eval()

    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:
        logger.info("Evaluating Test Set Pixels...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.test_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )
        
        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_test_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_test_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=2,
            verbose=True,
            save_images=False,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/test/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
        
    if cfg.render.render_full:
        logger.info("Evaluating Full Set...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )
        
        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/full/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            full_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_full_{current_time}.json"
            with open(full_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_img_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=cfg.render.fps,
            verbose=True,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/full/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    
    # render_novel_cfg = cfg.render.get("render_novel", None)
    # if render_novel_cfg is not None:
    #     logger.info("Rendering novel views...")
    #     render_traj = dataset.get_novel_render_traj(
    #         traj_types=render_novel_cfg.traj_types,
    #         target_frames=render_novel_cfg.get("frames", dataset.frame_num),
    #     )
    #     video_output_dir = f"{cfg.log_dir}/videos{post_fix}/novel_{step}"
    #     if not os.path.exists(video_output_dir):
    #         os.makedirs(video_output_dir)
        
    #     for traj_type, traj in render_traj.items():
    #         # Prepare rendering data
    #         render_data = dataset.prepare_novel_view_render_data(traj)
            
    #         # Render and save video
    #         save_path = os.path.join(video_output_dir, f"{traj_type}.mp4")
    #         render_novel_views(
    #             trainer, render_data, save_path,
    #             fps=render_novel_cfg.get("fps", cfg.render.fps)
    #         )
    #         logger.info(f"Saved novel view video for trajectory type: {traj_type} to {save_path}")

    if render_keys is not None and "single_wide" in render_keys:
        base_dir = output_root if output_root is not None else cfg.log_dir
        pseudo_wide_dir = os.path.join(base_dir, "pseudoGTWide")
        save_single_wide_rgbs(
            trainer=trainer,
            dataset=dataset,
            output_dir=pseudo_wide_dir,
            width_scale=0.5,
        )
            
def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    # Always evaluate into the checkpoint directory passed by --resume_from.
    # This avoids mismatches when cfg.log_dir in config.yaml points elsewhere.
    cfg.log_dir = log_dir
    args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )
    
    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        # "gt_rgbs",
        # "rgbs",
        "single_wide",
        # "Background_rgbs",
        # "RigidNodes_rgbs",
        # "DeformableNodes_rgbs",
        # "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False
    
    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix="_eval",
        output_root=log_dir,
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str, required=True)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
        
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    main(args)