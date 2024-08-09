import os

if os.getenv("MODE") == "PROFILE":
    import builtins
    import line_profiler
    import atexit

    profile = line_profiler.LineProfiler()
    atexit.register(profile.print_stats)
    builtins.__dict__["profile"] = profile
    print("Line profiler enabled")
