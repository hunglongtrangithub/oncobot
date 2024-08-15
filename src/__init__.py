import os

def enable_line_profiler():
    import builtins
    import line_profiler
    import atexit

    profile = line_profiler.LineProfiler()
    atexit.register(profile.print_stats)
    builtins.__dict__["profile"] = profile

if os.getenv("MODE") == "TEST":
    enable_line_profiler()
    print("Line profiler enabled")
