# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KITTI Odometry Dataset implementation.

KITTI is a popular dataset for autonomous driving. This implementation
focuses on the Odometry benchmark for camera pose estimation evaluation.
Reference: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

Evaluation metrics:
- Camera pose estimation: AUC metrics
- 3D reconstruction: Accuracy, Completeness, F-score (using LiDAR GT)
"""

import glob
import os
from typing import Dict as TDict, List, Optional

import cv2
import numpy as np
import open3d as o3d
from addict import Dict

from depth_anything_3.bench.dataset import Dataset, _wait_for_file_ready
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    evaluate_3d_reconstruction,
    fuse_depth_to_tsdf,
    sample_points_from_mesh,
)
from depth_anything_3.utils.constants import (
    KITTI_DOWN_SAMPLE,
    KITTI_DATA_ROOT,
    KITTI_POSE_ROOT,
    KITTI_EVAL_THRESHOLD,
    KITTI_MAX_DEPTH,
    KITTI_SAMPLING_NUMBER,
    KITTI_SCENES,
    KITTI_SDF_TRUNC,
    KITTI_VOXEL_LENGTH,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama


@MV_REGISTRY.register(name="kitti")
@MONO_REGISTRY.register(name="kitti")
class KITTIDataset(Dataset):
    """
    KITTI Odometry Dataset wrapper for DepthAnything3 evaluation.

    Supports:
        - Camera pose estimation evaluation (AUC metrics)
        - 3D reconstruction evaluation (Accuracy, Completeness, F-score)
        - TSDF-based point cloud fusion

    Dataset structure:
        KITTI_Odometry/
        ├── sequences/
        │   ├── 00/
        │   │   ├── image_2/           # Left color images
        │   │   ├── image_3/           # Right color images (optional)
        │   │   ├── velodyne/          # LiDAR scans (optional)
        │   │   └── calib.txt          # Calibration data
        │   └── ...
        └── poses/
            ├── 00.txt                 # Ground truth poses for sequence 00
            └── ...
    """

    data_root = KITTI_DATA_ROOT
    pose_root = KITTI_POSE_ROOT
    SCENES = KITTI_SCENES

    # Evaluation hyperparameters from constants
    max_depth = KITTI_MAX_DEPTH
    sampling_number = KITTI_SAMPLING_NUMBER
    voxel_length = KITTI_VOXEL_LENGTH
    sdf_trunc = KITTI_SDF_TRUNC
    eval_threshold = KITTI_EVAL_THRESHOLD
    down_sample = KITTI_DOWN_SAMPLE

    def __init__(self):
        super().__init__()
        self._scene_cache = {}

    # ------------------------------
    # Camera file parsing
    # ------------------------------

    def _parse_calib(self, filepath: str) -> dict:
        """Parse KITTI's calib.txt file."""
        calib = {}
        with open(filepath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                key, val = line.split(":", 1)
                calib[key.strip()] = np.array([float(x) for x in val.split()]).reshape(3, 4)
        return calib

    def _load_poses(self, filepath: str) -> np.ndarray:
        """Load KITTI's poses (flattened 3x4 C2W matrices)."""
        poses_raw = np.loadtxt(filepath)
        poses = poses_raw.reshape(-1, 3, 4)
        # Convert to 4x4
        p4x4 = np.tile(np.eye(4), (len(poses), 1, 1))
        p4x4[:, :3, :4] = poses
        return p4x4

    # ------------------------------
    # Public API
    # ------------------------------

    def get_data(self, scene: str) -> Dict:
        """
        Collect per-view image paths, intrinsics/extrinsics for a scene.

        Args:
            scene: Sequence ID (e.g., "00")

        Returns:
            Dict with:
                - image_files: List[str] - paths to images
                - extrinsics: np.ndarray [N, 4, 4] - world-to-camera transforms
                - intrinsics: np.ndarray [N, 3, 3] - camera intrinsics
                - aux: Dict with scene info and raw calib
        """
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = os.path.join(self.data_root, "sequences", scene)
        image_dir = os.path.join(scene_dir, "image_2")
        calib_file = os.path.join(scene_dir, "calib.txt")
        pose_file = os.path.join(self.pose_root, "poses", f"{scene}.txt")

        if not os.path.exists(image_dir) or not os.path.exists(calib_file) or not os.path.exists(pose_file):
            print(f"[KITTI] Warning: Scene {scene} data not found.")
            print(f"  Expected image_dir: {image_dir}")
            print(f"  Expected calib_file: {calib_file}")
            print(f"  Expected pose_file: {pose_file}")
            return Dict({
                "image_files": [],
                "extrinsics": np.zeros((0, 4, 4)),
                "intrinsics": np.zeros((0, 3, 3)),
                "aux": Dict(),
            })

        # Parse calib and poses
        calib = self._parse_calib(calib_file)
        # P2 is for image_2 (rectified)
        ixt = calib["P2"][:3, :3].astype(np.float32)
        
        # Determine baseline for camera 2 relative to camera 0
        # P_cam = K * [I | t] -> t = K^-1 * P[:, 3]
        # In KITTI, P2 = [f 0 cx | fx*bx]
        #               [0 f cy | 0]
        #               [0 0 1  | 0]
        # So bx = P2[0, 3] / P2[0, 0]
        bx = calib["P2"][0, 3] / calib["P2"][0, 0]
        t_0_to_2 = np.array([bx, 0, 0], dtype=np.float32)

        # Load poses (C2W for camera 0)
        c02ws = self._load_poses(pose_file)
        
        # Convert to C2W for camera 2
        # C2_w = C0_w * T_0_to_2
        c22ws = c02ws.copy()
        # Rotation is same for all rectified cameras
        c22ws[:, :3, 3] += np.matmul(c02ws[:, :3, :3], t_0_to_2)

        # Convert to W2C (extrinsics)
        extrinsics = np.stack([np.linalg.inv(p) for p in c22ws]).astype(np.float32)

        # Get all images
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        
        # Sync frames
        num_frames = min(len(image_files), len(extrinsics))
        image_files = image_files[:num_frames]
        extrinsics = extrinsics[:num_frames]
        intrinsics = np.stack([ixt.copy() for _ in range(num_frames)])

        out = Dict({
            "image_files": image_files,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "aux": Dict({
                "scene": scene,
                "calib": calib,
                "heights": [cv2.imread(image_files[0]).shape[0]] * num_frames if image_files else [],
                "widths": [cv2.imread(image_files[0]).shape[1]] * num_frames if image_files else [],
            }),
        })

        print(f"[KITTI] {scene}: {len(out.image_files)} images")
        self._scene_cache[scene] = out
        return out

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:
        """
        Evaluate fused point cloud against KITTI LiDAR ground truth.
        
        Note: This is a placeholder as KITTI doesn't provide a single mesh.
        Evaluation against raw LiDAR points for the sampled frames can be implemented.
        """
        # TODO: Implement evaluation against LiDAR scans if available
        print(f"[KITTI] 3D reconstruction evaluation for sequence {scene} is not fully implemented.")
        return {
            "acc": 0.0,
            "comp": 0.0,
            "overall": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
        }

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:
        """
        Fuse per-view depths into a point cloud using TSDF fusion.
        """
        # Try to load saved GT meta (handles frame sampling)
        gt_meta = self._load_gt_meta(result_path)
        if gt_meta is not None:
            gt_data = gt_meta
        else:
            gt_data = self.get_data(scene)
            
        _wait_for_file_ready(result_path)
        pred_data = Dict({k: v for k, v in np.load(result_path).items()})

        # Load images
        images = []
        orig_sizes = []
        for img_path in gt_data.image_files:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            orig_sizes.append((img.shape[0], img.shape[1]))

        # Prepare data with resize
        if mode == "recon_unposed":
            depths, intrinsics, extrinsics = self._prep_unposed(pred_data, gt_data, orig_sizes)
        elif mode == "recon_posed":
            depths, intrinsics, extrinsics = self._prep_posed(pred_data, gt_data, orig_sizes)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        images = np.stack(images, axis=0)

        # TSDF fusion
        volume = create_tsdf_volume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
        )
        mesh = fuse_depth_to_tsdf(
            volume, depths, images, intrinsics, extrinsics, max_depth=self.max_depth
        )

        # Sample and save
        pcd = sample_points_from_mesh(mesh, self.sampling_number)
        os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        o3d.io.write_point_cloud(fuse_path, pcd)

    def _load_gt_meta(self, result_path: str) -> Dict:
        """Load saved GT meta for fusion."""
        export_dir = os.path.dirname(result_path)
        gt_meta_path = os.path.join(os.path.dirname(export_dir), "gt_meta.npz")
        if os.path.exists(gt_meta_path):
            data = np.load(gt_meta_path, allow_pickle=True)
            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
                "image_files": list(data["image_files"]),
            })
        return None

    def _prep_unposed(self, pred_data: Dict, gt_data: Dict, orig_sizes: list) -> tuple:
        """Prepare for recon_unposed mode."""
        _, _, scale, extrinsics = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )
        model_h, model_w = pred_data.depth.shape[1], pred_data.depth.shape[2]
        depths_out = []
        intrinsics_out = []
        for i in range(len(pred_data.depth)):
            oh, ow = orig_sizes[i]
            depth = cv2.resize(pred_data.depth[i], (ow, oh), interpolation=cv2.INTER_NEAREST)
            depth = depth * scale
            ixt = pred_data.intrinsics[i].copy()
            ixt[0, :] *= (ow / model_w)
            ixt[1, :] *= (oh / model_h)
            depths_out.append(depth)
            intrinsics_out.append(ixt)
        return np.stack(depths_out), np.stack(intrinsics_out), extrinsics

    def _prep_posed(self, pred_data: Dict, gt_data: Dict, orig_sizes: list) -> tuple:
        """Prepare for recon_posed mode."""
        _, _, scale, _ = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )
        depths_out = []
        for i in range(len(pred_data.depth)):
            oh, ow = orig_sizes[i]
            depth = cv2.resize(pred_data.depth[i], (ow, oh), interpolation=cv2.INTER_NEAREST)
            depth = depth * scale
            depths_out.append(depth)
        return np.stack(depths_out), gt_data.intrinsics.copy(), gt_data.extrinsics.copy()
