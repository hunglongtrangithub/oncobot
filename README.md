# Requirements

- ffmpeg
- poetry
- yarn

# SadTalker:

```sh
cd src/sad_talker && chmod +x scripts/download_models.sh && ./scripts/download_models.sh
```

# Install:

```sh
poetry config virtualenvs.in-project true && poetry install # to set up virtual environment in project with the .venv folder
cd ui && yarn install
```

# Prepare Environment Variables:

1. Copy `.env.example` to `.env` and fill in the values
2. Fill in your `config.toml` file with the Meilisearch environment variables in `.env` file

# Start Meilisearch locally:

1. install Meilisearch
2. set environment variables in appropriate places in `.env.example`
3. run Meilisearch at project's root directory:

```sh
meilisearch --config-file-path="./config.toml"
```

# Start the application:
