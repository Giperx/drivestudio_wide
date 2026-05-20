from typing import List, Callable
import os
import gc
import joblib
import logging
import argparse
import numpy as np

from datasets.tools.extract_smpl import run_4DHumans
from datasets.tools.postprocess import match_and_postprocess

logger = logging.getLogger()


def cleanup_cuda_memory():
    """Best-effort GPU memory cleanup between scenes."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        # Keep cleanup best-effort and never block scene processing.
        pass


def parse_scene_ids_from_split(split_file_path: str, dataset: str):
    """Parse scene ids from split file.

    Supported formats:
    - plain ids per line, e.g. "002"
    - CSV first-column ids, e.g. "37,xxxx"
    - whitespace-separated first token ids
    """
    with open(split_file_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    lines = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError(f"Split file is empty: {split_file_path}")

    scene_ids = []
    for idx, line in enumerate(lines):
        token = line.split(",")[0].strip().split()[0]

        # Skip a common header row when present.
        if idx == 0 and token.lower() in {"scene", "scene_id", "id", "sceneid"}:
            continue

        if dataset == "kitti":
            # Keep KITTI ids as strings to preserve potential leading zeros or names.
            scene_ids.append(token)
            continue

        try:
            scene_ids.append(int(token))
        except ValueError as e:
            raise ValueError(
                f"Invalid scene id '{token}' in split file '{split_file_path}' for dataset '{dataset}'."
            ) from e

    if not scene_ids:
        raise ValueError(f"No valid scene ids parsed from split file: {split_file_path}")

    return scene_ids

def extract_humanpose(
    scene_dir,
    projection_fn: Callable,
    camera_list: List[str],
    save_temp: bool=True,
    verbose: bool=False,
    fps: int=12
):
    """Extract human pose from the waymo dataset
    
    Args:
        scene_dir: str, path to the scene directory
        save_temp: bool, whether to save the intermediate results
        verbose: bool, whether to visualize debug images
        fps: int, FPS for the visualization video
    """
    # project human boxes to 2D image space
    GTTracks_meta = projection_fn(
        scene_dir, camera_list=camera_list,
        save_temp=save_temp, verbose=verbose,
        narrow_width_ratio=0.2, fps=fps
    )
    
    # run 4DHuman to get predicted human tracks with SMPL parameters
    PredTracks_meta = run_4DHumans(
        scene_dir, camera_list=camera_list,
        save_temp=save_temp, verbose=verbose, fps=fps
    )
    
    # match the predicted tracks with the ground truth tracks
    smpl_meta = match_and_postprocess(
        scene_dir, camera_list=camera_list,
        GTTracksDict=GTTracks_meta, PredTracksDict=PredTracks_meta,
        save_temp=save_temp, verbose=verbose, fps=fps
    )
    
    joblib.dump(
        smpl_meta,
        os.path.join(scene_dir, "humanpose", "smpl.pkl")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument("--data_root", type=str, required=True, help="root path of waymo dataset")
    parser.add_argument("--dataset", type=str, default="waymo", help="dataset name")
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        "--numbers_idx",
        type=int,
        default=None,
        help="number of scenes to process from training list (overrides --num_scenes when --training_list is set)",
    )
    parser.add_argument(
        "--training_list",
        type=str,
        default="data/waymo_train_list.txt",
        help="path to training list file containing scene names, one per line",
    )
    parser.add_argument(
        "--save_temp",
        action="store_true",
        help="Whether to save the intermediate results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to visualize the intermediate results",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="FPS for the visualization video if verbose is True",
    )
    parser.add_argument(
        "--cleanup_cuda_per_scene",
        action="store_true",
        help="Force CUDA cache cleanup after each scene (recommended for NuScenes batch processing).",
    )
    parser.add_argument(
        "--retry_oom_without_verbose",
        action="store_true",
        help="If CUDA OOM occurs, retry the same scene once with verbose=False.",
    )
    args = parser.parse_args()
    
    if args.dataset == "waymo":
        from datasets.waymo.waymo_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "pandaset":
        from datasets.pandaset.pandaset_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "argoverse":
        from datasets.argoverse.argoverse_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "nuscenes":
        from datasets.nuscenes.nuscenes_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "kitti":
        from datasets.kitti.kitti_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "nuplan":
        from datasets.nuplan.nuplan_human_utils import project_human_boxes, CAMERA_LIST
    else:
        raise ValueError(f"Unknown dataset {args.dataset}, please choose from waymo, pandaset, argoverse, nuscenes, kitti, nuplan")
    
    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        scene_ids_list = parse_scene_ids_from_split(args.split_file, args.dataset)
    elif args.training_list is not None and os.path.exists(args.training_list):
        training_files = open(args.training_list).read().splitlines()
        training_files = [f.strip() for f in training_files if f.strip()]
        num = args.numbers_idx if args.numbers_idx is not None else args.num_scenes
        scene_ids_list = list(range(args.start_idx, args.start_idx + num))
        logger.info(f"Using training list: {len(scene_ids_list)} scenes [{args.start_idx}:{args.start_idx + num}]")
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    for scene_id in scene_ids_list:
        try:
            scene_dir = f'{args.data_root}/{str(scene_id).zfill(3)}'
            extract_humanpose(
                scene_dir=scene_dir,
                projection_fn=project_human_boxes,
                camera_list=CAMERA_LIST,
                save_temp=args.save_temp,
                verbose=args.verbose,
                fps=args.fps
            )
            logger.info(f"Finished processing scene {scene_id}")
        except Exception as e:
            err_msg = str(e)
            if args.retry_oom_without_verbose and ("out of memory" in err_msg.lower()):
                logger.warning(
                    f"CUDA OOM on scene {scene_id}. Retrying once with verbose=False."
                )
                cleanup_cuda_memory()
                try:
                    extract_humanpose(
                        scene_dir=scene_dir,
                        projection_fn=project_human_boxes,
                        camera_list=CAMERA_LIST,
                        save_temp=args.save_temp,
                        verbose=False,
                        fps=args.fps
                    )
                    logger.info(f"Finished processing scene {scene_id} after OOM retry")
                    continue
                except Exception as e_retry:
                    logger.error(f"Retry failed for scene {scene_id}: {e_retry}")
            logger.error(f"Error processing scene {scene_id}: {e}")
            continue
        finally:
            if args.cleanup_cuda_per_scene:
                cleanup_cuda_memory()