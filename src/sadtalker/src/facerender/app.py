from .animate import (
    AnimateFromCoeff,
    ACCEPTED_DTYPES,
    ACCEPTED_WEIGHTS,
    ACCEPTED_ACTIVATIONS,
)
from pathlib import Path
import subprocess
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
import torch
import time
import requests
from requests.exceptions import ConnectionError


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def check_arguments(torch_dtype, quanto_config):
    dtype = None
    if torch_dtype:
        if torch_dtype not in ACCEPTED_DTYPES:
            raise ValueError(
                f"Only support dtypes in {ACCEPTED_DTYPES.keys()} but found {torch_dtype}"
            )
        dtype = ACCEPTED_DTYPES[torch_dtype]
    if "weights" in quanto_config:
        if quanto_config["weights"] not in ACCEPTED_WEIGHTS:
            raise ValueError(
                f"Only support weights in {ACCEPTED_WEIGHTS.keys()} but found {quanto_config['weights']}"
            )
        quanto_config["weights"] = ACCEPTED_WEIGHTS[quanto_config["weights"]]
    if "activations" in quanto_config:
        if quanto_config["activations"] not in ACCEPTED_ACTIVATIONS:
            raise ValueError(
                f"Only support weights in {ACCEPTED_ACTIVATIONS.keys()} but found {quanto_config['activations']}"
            )
        quanto_config["activations"] = ACCEPTED_ACTIVATIONS[
            quanto_config["activations"]
        ]
    return dtype, quanto_config


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
            # print("Checking server status at {}".format(url))
            response = requests.get(url)
            if response.status_code == 200:
                return True  # Server is ready
        except ConnectionError:
            pass  # Server not yet ready
        time.sleep(1)  # Wait 1 second before retrying

    return False  # Server did not become ready within the timeout period


def run(port: int, host: str = "0.0.0.0", timeout: int = 10):
    # Start the server in a separate process
    cwd = Path(__file__).parents[4]
    process = subprocess.Popen(
        ["uvicorn", f"{__name__}:app", "--host", host, "--port", str(port)],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = "http://{}:{}".format(host, port)
    if not wait_for_server_ready(url, timeout):
        # If the server did not start within the timeout period, terminate the process and raise an exception
        process.terminate()
        raise Exception("Server did not start within the timeout period")

    # If the server started successfully, return the url
    return url
