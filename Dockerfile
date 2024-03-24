# Base image with CUDA support
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Python version
ENV PYTHON_VERSION=3.11

# Install Python, pip, and basic build utilities
RUN apt-get update && \
  apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip git build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
  update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install poetry==1.5.1

RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-interaction --no-ansi 

RUN poetry run python check_fastapi.py

WORKDIR /app

COPY ./*.py ./

COPY ./faiss_index ./faiss_index

# automating agreement to the terms and conditions of the coqui TTS model
COPY ./expect.exp ./
RUN chmod +x ./expect.exp

# COPY ./.env ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
