# Set-up

### Virtual Environment
- Using UV

### Packages
```
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch>=2" torchvision xformers
uv pip install -e .
```

### Commands
- `python3 get_poses.py /home/df/data/datasets/kitti/data_odometry_color/sequences/00/image_2 output_poses --max_images=100`
- ` python -m depth_anything_3.bench.evaluator data=[KITTI]` ?
### Docker
- `bash ./docker_build.sh` builds docker file
- `bash ./docker_run.sh` runs docker container, mounting current directory to easily access inputs/results
- Run commands