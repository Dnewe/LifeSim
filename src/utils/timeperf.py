import time
from functools import wraps
from collections import defaultdict

TIMINGS = defaultdict(float)

beginning = time.perf_counter()

def timed(name=None):
    def decorator(func):
        key = name or func.__qualname__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            TIMINGS[key] += time.perf_counter() - start
            return result

        return wrapper
    return decorator


def print_relative():
    total = sum(TIMINGS.values())
    print("Performance time (relative)")
    for name, t in TIMINGS.items():
        print(f"    {name}: {t / total:.2%}")
        

def print_global():
    total = time.perf_counter() - beginning
    print("Performance time (global)")
    for name, t in TIMINGS.items():
        print(f"    {name}: {t / total:.2%}")
    