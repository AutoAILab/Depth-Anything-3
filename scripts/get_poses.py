import glob
import os
import torch
import numpy as np
import argparse
from depth_anything_3.api import DepthAnything3

def save_kitti_odometry(extrinsics, output_path):
    """
    Saves extrinsics in KITTI odometry format.
    extrinsics: (N, 3, 4) or (N, 4, 4) W2C matrices (OpenCV format)
    KITTI format: N lines, each with 12 numbers (3x4 C2W matrix flattened)
    """
    poses_kitti = []
    
    # 1. Convert to 4x4 if necessary and invert to get C2W
    num_poses = len(extrinsics)
    c2ws = []
    for i in range(num_poses):
        w2c = extrinsics[i]
        if w2c.shape == (3, 4):
            # Pad to 4x4
            w2c_4x4 = np.eye(4)
            w2c_4x4[:3, :4] = w2c
        else:
            w2c_4x4 = w2c
            
        # C2W = inverse(W2C)
        c2w = np.linalg.inv(w2c_4x4)
        c2ws.append(c2w)
    
    # 2. Normalize such that the first pose is identity
    # P_i_rel = inv(C2W_0) @ C2W_i
    first_c2w_inv = np.linalg.inv(c2ws[0])
    
    for i in range(num_poses):
        c2w_rel = first_c2w_inv @ c2ws[i]
        
        # Flatten the 3x4 part
        pose_12 = c2w_rel[:3, :4].reshape(-1)
        poses_kitti.append(pose_12)
    
    # Save to file
    with open(output_path, 'w') as f:
        for pose in poses_kitti:
            line = " ".join([f"{x:.6e}" for x in pose])
            f.write(line + "\n")
    print(f"Saved {len(poses_kitti)} poses to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Get camera poses from a set of images using Depth Anything 3.")
    parser.add_argument("image_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output.")
    parser.add_argument("--model", type=str, default="depth-anything/DA3-LARGE-1.1", help="Model name to use.")
    parser.add_argument("--use_ray_pose", action="store_true", default=True, help="Use ray-based xpose estimation (more accurate).")
    parser.add_argument("--max_images", type=int, default=-1, help="Maximum number of images to process (-1 for all).")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sampling images (e.g., 2 to process every second image).")
    parser.add_argument("--export_format", type=str, default=None, help="Export format (e.g., 'glb').")
    
    args = parser.parse_args()

    # 1. Initialize the model
    print(f"Loading model: {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model).to(device)

    # 2. Prepare your image list
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.[jp][pn][g]"))) # Handles .jpg, .png, etc.

    if args.stride > 1:
        image_paths = image_paths[::args.stride]
        print(f"Sampling with stride {args.stride}: {len(image_paths)} images remaining.")

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
        print(f"Limiting to first {args.max_images} images.")

    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Processing {len(image_paths)} images from {args.image_dir}...")

    # 3. Run inference
    prediction = model.inference(
        image_paths, 
        use_ray_pose=args.use_ray_pose,
        export_dir=args.output_dir,
        export_format=args.export_format
    )

    # 4. Extract and save the poses
    extrinsics = prediction.extrinsics # (N, 4, 4) or (N, 3, 4)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
        
    output_file = os.path.join(args.output_dir, "poses_kitti.txt")
    save_kitti_odometry(extrinsics, output_file)


    print(f"Extrinsics shape: {extrinsics.shape}")
    print(f"Intrinsics shape: {prediction.intrinsics.shape}")

if __name__ == "__main__":
    main()