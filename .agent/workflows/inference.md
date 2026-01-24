---
description: Perform basic depth estimation using the CLI or Python API
---

### CLI Usage

1. **Auto Mode (Batch Processing)**
   ```bash
   da3 auto [INPUT_DIR/IMAGE_PATH] --model-dir [MODEL_DIR] --export-format glb --export-dir [OUTPUT_DIR]
   ```

2. **Video Processing**
   ```bash
   da3 video [VIDEO_PATH] --fps 15 --export-format glb --export-dir [OUTPUT_DIR]
   ```

3. **Backend Reuse (Faster for multiple runs)**
   ```bash
   da3 backend --model-dir [MODEL_DIR] --gallery-dir [GALLERY_DIR]
   da3 auto [INPUT_DIR] --use-backend --export-dir [OUTPUT_DIR]
   ```

---

### Python API Usage

```python
import glob, os, torch
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

images = sorted(glob.glob("path/to/images/*.png"))
prediction = model.inference(images)

print(f"Depth shape: {prediction.depth.shape}")
```
