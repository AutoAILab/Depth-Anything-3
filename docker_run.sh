# remove container and image
docker container remove dp3-env

# run in interactive mode using bash
    # mount directory to run files
    # mount data drive to access datasets
docker run \
  --mount type=bind,src="/home/df/data/jflinte/Depth-Anything-3",dst=/app \
  --mount type=bind,src="/home/df/data",dst=/app/data \
  -it --gpus all --name dp3-env dp3-env bash

