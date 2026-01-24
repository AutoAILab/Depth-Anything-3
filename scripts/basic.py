# imports
import glob
import os
import torch
from depth_anything_3.api import DepthAnything3

# constants
MAX_IMAGES = 50

# set up model
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

# example_path = "/app/data/datasets/kitti/data_odometry_color/sequences/01/image_2/"
example_path = "./assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
images = images[:50]  # take only first 50 images

print(f"{len(images)} image/s found")


# infer
prediction = model.inference(
    images,  # type: ignore
    export_dir='assets/test',
    export_format='glb-npz'

)

# print predictions
# prediction.processed_images : [N, H, W, 3] uint8   array
if prediction.processed_images is not None:
    print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)
# prediction.conf             : [N, H, W]    float32 array
if prediction.conf is not None:
    print(prediction.conf.shape)
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
if prediction.extrinsics is not None:
    print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
if prediction.intrinsics is not None:
    print(prediction.intrinsics.shape)

print(prediction.extrinsics)