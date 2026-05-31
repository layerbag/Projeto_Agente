import time
import asyncio
from functools import wraps
# pyrefly: ignore [missing-import]
from src.mlops.logging import logger

def measure_time(name: str):
    def decorator(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            logger.info(f"{name} took {time.time() - start:.2f}s")
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            logger.info(f"{name} took {time.time() - start:.2f}s")
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

