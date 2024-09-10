import os
from src.utils.logger_config import logger

if os.getenv("MODE") == "profile":
    import builtins
    import line_profiler
    import atexit

    profile = line_profiler.LineProfiler()
    atexit.register(profile.print_stats)
    builtins.__dict__["profile"] = profile
    logger.info("Line profiler enabled")
