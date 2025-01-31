# Define variables for common parameters
IMAGE_NAME=chat-frontend
CONTAINER_NAME=chat-frontend
PORT=3000
WORKDIR=/usr/src/app
LOG?=
.PHONY: build dev stop remove rebuild

# Build the Docker image
build:
	docker build $(if $(findstring $(LOG), 1),--no-cache --progress=plain,) -t $(IMAGE_NAME) .


# Run the Docker container
# --mount type=bind,source=$(shell realpath .),target=$(WORKDIR) \

dev:
	docker run --rm --name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		--entrypoint "/bin/bash" \
		-w $(WORKDIR) \
		$(IMAGE_NAME) \
		-c "bun run dev"


# Interactive shell into the Docker CONTAINER_NAME
# --mount type=bind,source=$(shell realpath .),target=$(WORKDIR) \

shell:
	docker run --rm -it --name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		--entrypoint "/bin/bash" \
		-w $(WORKDIR) \
		$(IMAGE_NAME) \
		-c "/bin/bash"

# Stop the Docker container
stop:
	docker stop $(CONTAINER_NAME)

# Start the Docker container
start:
	docker start $(CONTAINER_NAME)

# Remove the Docker container
remove:
	docker rm $(CONTAINER_NAME)

# Rebuild and restart the container
rebuild: stop remove build

