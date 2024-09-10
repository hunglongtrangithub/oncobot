import os
from pathlib import Path
from loguru import logger

MODE = os.getenv("MODE", "development")

if MODE == "production":
    log_file = Path("./log/oncobot.log")
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, level="INFO", rotation="10 MB")
