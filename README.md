# Oncobot: A Self-Hosted Multimodal Chatbot for Enhanced Clinical Oncology Information Retrieval

## Requirements

- ffmpeg
- uv
- bun

## SadTalker:

```sh
cd src/sadtalker && chmod +x scripts/download_models.sh && ./scripts/download_models.sh
```

## Install:

```sh
uv sync && source .venv/bin/activate
cd ui && yarn install
```

## Prepare Environment Variables:

1. Rename `.env.example` to `.env` and fill in your environment variables
2. Rename `config.toml.example` to `config.toml` and fill in the Meilisearch environment variables in `.env` file
3. Make sure your Meilisearch master key is more than 16 bytes

## Start Meilisearch locally:

1. install Meilisearch
3. run Meilisearch at project's root directory:

```sh
meilisearch --config-file-path="./config.toml"
```

4. run the following command to index the data:

```sh
python -m src.oncobot.scripts.index
```

## Start the servers in development mode:

### Start the FastAPI server:

```sh
make dev
```

### Start the Next.js server:

```sh
cd ui && bun run dev
```

## TODO

1. need a lightweight Ollama moddel to test integrating that model being served locally with Oncobot
2. If 1 works, then try with vllm
3. If 2 works without problems, try adding chat template and inference parameters to Vllm.
