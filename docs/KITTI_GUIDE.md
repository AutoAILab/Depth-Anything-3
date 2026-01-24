# KITTI Processing Guide

This guide describes how to process KITTI Odometry sequences using Depth Anything 3 to generate camera poses, evaluate accuracy, and create 3D reconstructions.

## 🚀 Unified Pipeline

The most convenient way to process a KITTI sequence is using the `scripts/kitti_pipeline.py` script.

```bash
# Process sequence 00 (limit to 100 frames for testing)
python scripts/kitti_pipeline.py --sequence 00 --output-dir output/kitti_00 --max-images 100

# Process full sequence 05
python scripts/kitti_pipeline.py --sequence 05 --output-dir output/kitti_05 --max-images -1
```

### Pipeline Steps:
1. **Pose Inversion & Reconstruction**: Runs `get_poses.py` to estimate camera poses and export a `.glb` file for 3D visualization.
2. **Evaluation**: Runs `evaluate_poses.py` to compare estimated poses with ground truth, calculating AUC@30, RRE, and RTE metrics.
3. **Visualization**: Generates accuracy curves and organizes results for Gradio integration.

## 📈 Evaluation Metrics

The pipeline outputs:
- `poses_kitti.txt`: Estimated camera poses in KITTI format.
- `pose_auc30.png`: Plot of the accuracy-under-threshold curve.
- `scene.glb`: 3D reconstruction of the sequence.

You can view evaluation results in the terminal:
```text
========================================
POSE EVALUATION RESULTS (Threshold: 30°)
========================================
AUC@30: 0.952694
Mean RRE: 1.0032°
Mean RTE (angular): 1.6847°
========================================
```

## 🎨 Gradio Integration

To view the results in the interactive web UI:

1. Move the output folder to the gallery directory:
   ```bash
   mkdir -p workspace/gallery
   mv output/kitti_00 workspace/gallery/
   ```
2. Launch the Gradio app:
   ```bash
   da3 gradio
   ```
3. Navigate to the "Gallery" tab in the web interface to see the KITTI sequence reconstruction.

## 🛠️ Manual CLI Usage

If you prefer to run steps individually:

### 1. Get Poses & GLB
```bash
python scripts/get_poses.py /path/to/kitti/images output/dir --export_format glb
```

### 2. Evaluate
```bash
python scripts/evaluate_poses.py output/dir/poses_kitti.txt /path/to/kitti/gt_poses.txt --threshold 30 --plot
```

---
**Note**: Ensure you have activated the virtual environment (`source .venv/bin/activate`) before running these commands.
