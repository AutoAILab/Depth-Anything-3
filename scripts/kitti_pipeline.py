import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        # sys.exit(result.returncode)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="KITTI Pose Pipeline Orchestrator")
    parser.add_argument("--sequence", type=str, required=True,
                       help="KITTI sequence number (e.g., '00')")
    parser.add_argument("--kitti-dir", type=str, default="/home/df/data/datasets/kitti",
                       help="Path to KITTI dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--model", type=str, default="depth-anything/DA3-LARGE-1.1",
                       help="Model name to use")
    parser.add_argument("--max-images", type=int, default=100,
                       help="Maximum number of images to process (default: 100 for testing)")
    parser.add_argument("--stride", type=int, default=1,
                       help="Stride for sampling images")

    args = parser.parse_args()

    # Paths
    kitti_path = Path(args.kitti_dir)
    image_dir = kitti_path / "data_odometry_color" / "sequences" / args.sequence / "image_2"
    gt_file = kitti_path / "data_odometry_poses" / "poses" / f"{args.sequence}.txt"
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)
    if not gt_file.exists():
        print(f"Error: GT file not found: {gt_file}")
        sys.exit(1)

    print(f"--- Phase 2: Pose Estimation & Reconstruction ---")
    get_poses_cmd = [
        "python3", "scripts/get_poses.py",
        str(image_dir),
        str(output_path),
        "--model", args.model,
        "--max_images", str(args.max_images),
        "--stride", str(args.stride),
        "--use_ray_pose",
        "--export_format", "glb"
    ]
    run_command(get_poses_cmd)

    print(f"\n--- Phase 3: Evaluation ---")
    pred_file = output_path / "poses_kitti.txt"
    eval_cmd = [
        "python3", "scripts/evaluate_poses.py",
        str(pred_file),
        str(gt_file),
        "--threshold", "30",
        "--plot",
        "--output_plot", str(output_path / "pose_auc30.png")
    ]
    run_command(eval_cmd)

    print(f"\nPipeline execution completed for sequence {args.sequence}")
    print(f"Results located in: {output_path}")


if __name__ == "__main__":
    main()