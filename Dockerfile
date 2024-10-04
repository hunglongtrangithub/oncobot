FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Install ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Install Rust compiler
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
  && . $HOME/.cargo/env
# Ensure the Rust compiler is in the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:0.4.1 /uv /bin/uv

WORKDIR /app

COPY ./pyproject.toml ./uv.lock ./
RUN uv sync --frozen --no-cache
RUN . .venv/bin/activate

COPY ./src/ ./src/

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
