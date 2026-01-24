import numpy as np
import torch
import argparse
import os
import matplotlib.pyplot as plt
from depth_anything_3.bench.utils import compute_pose, calculate_auc_np, se3_to_relative_pose_error, align_to_first_camera

def load_kitti_poses(file_path):
    """
    Load poses from a KITTI format text file.
    Each line is 12 numbers: r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            data = list(map(float, line.split()))
            if len(data) == 12:
                pose = np.eye(4)
                pose[:3, :4] = np.array(data).reshape(3, 4)
                poses.append(pose)
    return np.array(poses)

def plot_accuracy_curve(errors, max_threshold, label, output_path):
    """
    Plots the accuracy-under-threshold curve.
    """
    thresholds = np.linspace(0, max_threshold, 100)
    accuracies = []
    for t in thresholds:
        acc = np.mean(errors < t)
        accuracies.append(acc)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, accuracies, label=f"{label} (AUC: {np.mean(accuracies):.4f})")
    plt.xlabel("Error Threshold (degrees)")
    plt.ylabel("Accuracy")
    plt.title(f"Pose Accuracy Curve (up to {max_threshold}°)")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    print(f"Curve plotted and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate camera poses against ground truth.")
    parser.add_argument("pred_file", type=str, help="Path to predicted poses (KITTI format).")
    parser.add_argument("gt_file", type=str, help="Path to ground truth poses (KITTI format).")
    parser.add_argument("--threshold", type=int, choices=[3, 30], default=3, help="AUC threshold (3 or 30).")
    parser.add_argument("--plot", action="store_true", help="Plot the accuracy curve.")
    parser.add_argument("--output_plot", type=str, default="accuracy_curve.png", help="Filename for the plotted curve.")

    args = parser.parse_args()

    # 1. Load poses
    print(f"Loading predicted poses from {args.pred_file}...")
    pred_poses = load_kitti_poses(args.pred_file)
    print(f"Loading ground truth poses from {args.gt_file}...")
    gt_poses = load_kitti_poses(args.gt_file)

    if len(pred_poses) != len(gt_poses):
        print(f"Warning: Number of poses mismatch! Pred: {len(pred_poses)}, GT: {len(gt_poses)}")
        min_len = min(len(pred_poses), len(gt_poses))
        pred_poses = pred_poses[:min_len]
        gt_poses = gt_poses[:min_len]

    # 2. Convert to tensors for bench utils
    pred_se3 = torch.from_numpy(pred_poses).float()
    gt_se3 = torch.from_numpy(gt_poses).float()

    # 3. Calculate Relative Rotation and Translation Errors (all pairs)
    print("Calculating relative pose errors (pairwise)...")
    # We need to align to first camera first as per DA3 evaluation protocol
    pred_se3 = align_to_first_camera(pred_se3)
    gt_se3 = align_to_first_camera(gt_se3)
    
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, len(pred_se3))
    
    r_error = rel_rangle_deg.cpu().numpy()
    t_error = rel_tangle_deg.cpu().numpy()
    
    # Pose error is the max of R and T errors
    max_errors = np.max(np.stack([r_error, t_error], axis=1), axis=1)

    # 4. Calculate AUC
    auc_val, _ = calculate_auc_np(r_error, t_error, max_threshold=args.threshold)
    
    print("\n" + "="*40)
    print(f"POSE EVALUATION RESULTS (Threshold: {args.threshold}°)")
    print("="*40)
    print(f"AUC@{args.threshold}: {auc_val:.6f}")
    print(f"Mean RRE: {np.mean(r_error):.4f}°")
    print(f"Mean RTE (angular): {np.mean(t_error):.4f}°")
    print("="*40)

    # 5. Plot if requested
    if args.plot:
        plot_accuracy_curve(max_errors, args.threshold, f"AUC@{args.threshold}", args.output_plot)

if __name__ == "__main__":
    main()

    #https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/evaluator.py