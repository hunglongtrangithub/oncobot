# Oncobot: A Self-Hosted Multimodal Chatbot for Enhanced Clinical Oncology Information Retrieval

## Requirements

- ffmpeg
- uv
- bun

## SadTalker:

Install `wget` if not already installed, then run the following command to download the pre-trained model weights for SadTalker:

```sh
cd src/sadtalker && chmod +x scripts/download_models.sh && ./scripts/download_models.sh && cd ../..
```

## Install:

Install dependencies for Python side:

```sh
uv sync
```

Install dependencies for Next.js side:

```sh
cd ui && bun install && cd ..
```

## Prepare Environment Variables:

1. Rename `.env.example` to `.env` and configure your environment variables for LLM API key(s)
2. Rename `config.toml.example` to `config.toml` and fill in the master key. Make sure your Meilisearch master key is more than 16 bytes (16 ASCII characters) for security

## Start Meilisearch locally:

1. install Meilisearch
3. run Meilisearch at project's root directory:

```sh
meilisearch --config-file-path="./config.toml"
```

4. run the following command to index the data (remember to activate your virtual environment first)

```sh
python -m src.oncobot.scripts.index
```

## Start the servers in development mode:

### Start the FastAPI server:

```sh
make dev
```

Use testing mode (`MODE=testing`) to start the server with dummy models:
```sh
MODE=testing make dev
```

### Start the Next.js server:

```sh
cd ui && bun run dev
```

### Modes in the backend:
Configured by the `MODE` environment variable, there are 4 modes for the backend server:
1. `production`: logs to file `./log/oncobot.log` with level `INFO`
2. `development`: default mode, logs to console with level `DEBUG`
3. `testing`: logs to console with level `DEBUG`, and uses dummy models for LLM and image captioning, which return fixed responses. This mode is for testing the integration of the frontend and backend without needing real models.
4. `profile`: enables `line_profiler` for profiling SadTalker.

## TODO

1. need a lightweight Ollama moddel to test integrating that model being served locally with Oncobot
2. If 1 works, then try with vllm
3. If 2 works without problems, try adding chat template and inference parameters to Vllm.
