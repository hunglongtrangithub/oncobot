# SadTalker:
cd src/sad_talker && chmod +x scripts/download_models.sh && ./scripts/download_models.sh

# Install:
```sh
poetry config virtualenvs.in-project true # to set up virtual environment in project with the .venv folder
poetry install
```

# Start Meilisearch locally:
1. install meilisearch
2. set environment variables in appropriate places in `.env.example`
3. run meilisearch at project's root dir:
```sh
meilisearch --config-file-path="./config.toml"
```
