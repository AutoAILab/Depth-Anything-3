# imports
import cv2
import glob
import os
from typing import List
import torch
from depth_anything_3.api import DepthAnything3


def jpg_to_png(images: List[str]) -> List[str]:
    """
    Converts list of jpg images to a list of png images
    """
    png_images: List[str] = []
    for img_path in images:
        if not img_path.lower().endswith('.jpg'):
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        png_path = img_path.rsplit('.', 1)[0] + '.png'
        success = cv2.imwrite(png_path, img)
        if success:
            png_images.append(png_path)
        else:
            print(f"Failed to save {png_path}")
    return png_images

# constants
MAX_IMAGES = 50
IMAGE_TYPE = 'jpg'

# set up model
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

# Images
example_path = "/home/df/data/jflinte/datasets/drone_dataset_brighton_beach/images"
# example_path = "/app/data/datasets/kitti/data_odometry_color/sequences/01/image_2/"

if IMAGE_TYPE == 'jpg':
    images = sorted(glob.glob(os.path.join(example_path, "*.JPG")))
    images = jpg_to_png(images)
elif IMAGE_TYPE == 'png':
    images = sorted(glob.glob(os.path.join(example_path, "*.png")))


images = images[:MAX_IMAGES]  # take only first 50 images

print(f"{len(images)} image/s found")


# infer
prediction = model.inference(
    images,  # type: ignore
    export_dir='/home/df/data/jflinte/Depth-Anything-3/assets/brighten_beach',
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