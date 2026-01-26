import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# -----------------------------
# Loading utilities
# -----------------------------

def load_estimated_poses(path):
    """
    Each line: 16 floats (4x4 row-major)
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = np.fromstring(line, sep=' ')
            assert len(vals) == 16
            T = vals.reshape(4, 4)
            poses.append(T)
    return np.array(poses)

def load_kitti_gt_poses(path):
    """
    Each line: 12 floats (3x4 row-major)
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = np.fromstring(line, sep=' ')
            assert len(vals) == 12
            T = np.eye(4)
            T[:3, :4] = vals.reshape(3, 4)
            poses.append(T)
    return np.array(poses)

def extract_positions(poses):
    """
    poses: (N, 4, 4)
    returns: (N, 3)
    """
    return poses[:, :3, 3]

# -----------------------------
# Umeyama alignment (Sim(3))
# -----------------------------

def umeyama_alignment(x, y, with_scale=True):
    """
    Align x to y using Umeyama method.
    x, y: (N, 3)
    Returns: s, R, t such that:
        y ≈ s * R @ x + t
    """
    assert x.shape == y.shape

    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)

    x_centered = x - mean_x
    y_centered = y - mean_y

    cov = x_centered.T @ y_centered / x.shape[0]

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = Vt.T @ S @ U.T

    if with_scale:
        var_x = np.var(x_centered, axis=0).sum()
        scale = np.trace(np.diag(D) @ S) / var_x
    else:
        scale = 1.0

    t = mean_y - scale * R @ mean_x

    return scale, R, t

def apply_alignment(x, s, R, t):
    return (s * (R @ x.T)).T + t

# -----------------------------
# ATE computation
# -----------------------------

def compute_ate(gt, est_aligned):
    """
    RMSE of Euclidean distance
    """
    errors = np.linalg.norm(gt - est_aligned, axis=1)
    return np.sqrt(np.mean(errors ** 2))

# -----------------------------
# Plotting
# -----------------------------

def plot_trajectories(gt, est, out_path):
    plt.figure(figsize=(8, 8))
    plt.plot(gt[:, 0], gt[:, 2], label="Ground Truth", linewidth=2)
    plt.plot(est[:, 0], est[:, 2], label="Estimated (aligned)", linestyle='--')
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.title("Trajectory Comparison (KITTI 00)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -----------------------------
# Main
# -----------------------------

def main(args):
    gt_poses = load_kitti_gt_poses(args.gt)
    est_poses = load_kitti_gt_poses(args.est)
    # est_poses = load_estimated_poses(args.est)

    assert len(gt_poses) == len(est_poses)

    gt_xyz = extract_positions(gt_poses)
    est_xyz = extract_positions(est_poses)

    # Align estimated to GT
    s, R, t = umeyama_alignment(est_xyz, gt_xyz, with_scale=True)
    est_xyz_aligned = apply_alignment(est_xyz, s, R, t)

    ate = compute_ate(gt_xyz, est_xyz_aligned)

    print("========== Evaluation ==========")
    print(f"Number of poses: {len(gt_xyz)}")
    print(f"Scale factor: {s:.6f}")
    print(f"ATE (RMSE): {ate:.4f} meters")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_path = os.path.join(args.out_dir, "trajectory.png")
    plot_trajectories(gt_xyz, est_xyz_aligned, plot_path)

    print(f"Trajectory plot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground truth pose file")
    parser.add_argument("--est", required=True, help="Estimated pose file")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    main(args)
    
    # python3 scripts/trajectory.py --gt "/home/df/data/datasets/kitti/data_odometry_poses/poses/04.txt" --est "output_poses/04/poses_kitti.txt" --out_dir "output_poses/04/"
    