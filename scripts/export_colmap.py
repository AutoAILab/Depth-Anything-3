import os
import numpy as np
import torch
import cv2
from addict import Dict
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export.colmap import export_to_colmap

def main(scene_path):
    # 1. Load predicted data and ground truth metadata
    results_path = os.path.join(scene_path, "exports/mini_npz/results.npz")
    meta_path = os.path.join(scene_path, "exports/gt_meta.npz")
    colmap_output_dir = os.path.join(scene_path, "exports/colmap")
    
    print(f"Loading results from {results_path}...")
    pred = np.load(results_path)
    meta = np.load(meta_path, allow_pickle=True)
    
    image_paths = meta['image_files'].tolist()
    
    # 2. Load images for COLMAP colors
    processed_images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_images.append(img)
    
    # 3. Resize depth and confidence to original image size
    num_frames = len(image_paths)
    orig_h, orig_w = processed_images[0].shape[:2]
    model_h, model_w = pred['depth'].shape[1:3]
    
    resized_depth = []
    resized_conf = []
    adjusted_intrinsics = []
    
    for i in range(num_frames):
        # Resize using nearest to preserve depth values
        d = cv2.resize(pred['depth'][i], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        c = cv2.resize(pred['conf'][i], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        resized_depth.append(d)
        resized_conf.append(c)
        
        # Adjust intrinsics
        ixt = pred['intrinsics'][i].copy()
        ixt[0, :] *= (orig_w / model_w)
        ixt[1, :] *= (orig_h / model_h)
        adjusted_intrinsics.append(ixt)

    prediction = Prediction(
        depth=np.stack(resized_depth),
        is_metric=1,
        intrinsics=np.stack(adjusted_intrinsics),
        extrinsics=pred['extrinsics'],
        conf=np.stack(resized_conf),
        processed_images=np.stack(processed_images)
    )
    
    # 4. Export to COLMAP format
    os.makedirs(colmap_output_dir, exist_ok=True)
    export_to_colmap(
        prediction=prediction,
        export_dir=colmap_output_dir,
        image_paths=image_paths,
        conf_thresh_percentile=40.0
    )
    print(f"Successfully exported to {colmap_output_dir}")

if __name__ == "__main__":
    # Example for KITTI sequence 00
    scene_dir = "workspace/evaluation/model_results/kitti/00/unposed"
    main(scene_dir)