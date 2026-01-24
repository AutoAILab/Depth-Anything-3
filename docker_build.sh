# remove container and image
docker container remove dp3-env
docker image remove dp3-env

# build image
docker build -t dp3-env .