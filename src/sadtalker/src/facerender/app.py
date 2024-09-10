from pathlib import Path
import subprocess
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
import torch
import time
import requests
from requests.exceptions import ConnectionError
import atexit

from .animate import check_arguments
from src.utils.logger_config import logger

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class SetUpRequest(BaseModel):
    sadtalker_paths: dict[str, str | bool]
    device: str | int
    dtype: Optional[str]
    dp_device_ids: Optional[list[int]]
    quanto_config: dict[str, str]


model = None


@app.post("/setup")
def setup(request: SetUpRequest):
    # Check if the model is already set in app state
    if hasattr(app.state, "model") and app.state.model is not None:
        return {"success": False, "message": "Model is already initialized."}
    dtype, quanto_config = check_arguments(request.dtype, request.quanto_config)
    device = torch.device(request.device)
    app.state.model = AnimateFromCoeff(
        request.sadtalker_paths,
        device=device,
        dtype=dtype,
        dp_device_ids=request.dp_device_ids,
        **quanto_config,
    )
    return {"success": True}


class PredictRequest(BaseModel):
    x: dict[str, list | int]


@app.post("/predict")
def predict(request: PredictRequest):
    if not hasattr(app.state, "model") or app.state.model is None:
        return {"error": "Model is not initialized."}, 400

    model = app.state.model

    x = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in request.x.items()}
    preiction_tensor = model.generate(x)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {"prediction": preiction_tensor.tolist()}


def wait_for_server_ready(url, timeout=10):
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            logger.info("Checking server status at {}".format(url))
            response = requests.get(url)
            if response.status_code == 200:
                return True  # Server is ready
        except ConnectionError:
            pass  # Server not yet ready
        time.sleep(1)  # Wait 1 second before retrying

    return False  # Server did not become ready within the timeout period


def terminate_process(process):
    logger.info("Terminating the server process")
    if process.poll() is None:  # Check if the process is still running
        process.terminate()
        try:
            process.wait(timeout=5)  # Wait for the process to terminate
        except subprocess.TimeoutExpired:
            process.kill()  # Force kill if it doesn't terminate in time


def run(port: int, host: str = "0.0.0.0", timeout: int = 10):
    # Start the server in a separate process
    cwd = Path(__file__).parents[4]
    process = subprocess.Popen(
        ["uvicorn", f"{__name__}:app", "--host", host, "--port", str(port)],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Register the cleanup function to terminate the subprocess on exit
    atexit.register(terminate_process, process)

    url = "http://{}:{}".format(host, port)
    if not wait_for_server_ready(url, timeout):
        # If the server did not start within the timeout period, terminate the process and raise an exception
        process.terminate()
        raise Exception("Server did not start within the timeout period")

    # If the server started successfully, return the url
    return url
