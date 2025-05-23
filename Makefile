# Define variables for common parameters
IMAGE_NAME=chat-backend
CONTAINER_NAME=chat-backend
PORT=8080
NO_GPU?=
LOG?=
.PHONY: start format build run stop remove rebuild test

# Start MeiliSearch
meilisearch:
	meilisearch --config-file-path="./config.toml"


# Start the local server using uvicorn
dev:
	uvicorn src.main:app --reload --port $(PORT)

# Start the local test server with uvicorn
test:
	MODE=TEST uvicorn src.main:app --reload --port $(PORT)

# Format the code using black and isort
format:
	black .
	isort .

# Build the Docker image
build:
	docker build $(if $(findstring $(LOG), 1),--no-cache --progress=plain,) -t $(IMAGE_NAME) .


# Run the Docker container
run:
	docker run --name $(CONTAINER_NAME) \
		$(if $(findstring $(NO_GPU), 1),,--gpus all) \
		--entrypoint "/bin/bash" \
		-p $(PORT):$(PORT) \
		-e NVIDIA_VISIBLE_DEVICES=all \
		--env-file .env \
		--mount type=bind,source=$(shell pwd),target=/app \
		--mount type=bind,source=$(shell pwd)/voices,target=/app/voices,readonly \
		--mount type=bind,source=$(shell pwd)/faiss_index,target=/app/faiss_index,readonly \
		--mount type=bind,source=$(shell pwd)/llm_llama,target=/app/llm_llama,readonly \
		$(IMAGE_NAME) \
		-c "uvicorn main:app --host 0.0.0.0 --port $(PORT) --reload"


# Interactive shell into the Docker CONTAINER_NAME
shell:
	docker run -it --name $(CONTAINER_NAME) \
		$(if $(findstring $(NO_GPU), 1),,--gpus all) \
		--entrypoint "/bin/bash" \
		-p $(PORT):$(PORT) \
		-e NVIDIA_VISIBLE_DEVICES=all \
		--env-file .env \
		--mount type=bind,source=$(shell pwd),target=/app \
		--mount type=bind,source=$(shell pwd)/voices,target=/app/voices,readonly \
		--mount type=bind,source=$(shell pwd)/faiss_index,target=/app/faiss_index,readonly \
		--mount type=bind,source=$(shell pwd)/llm_llama,target=/app/llm_llama,readonly \
		-m 12g \
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
rebuild: 
	stop remove build run

