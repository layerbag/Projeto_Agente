import time
from functools import wraps
from src.mlops.logging import logger

def measure_time(name: str):
    def decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):

            start = time.time()

            result = func(*args, **kwargs)

            ellapsed = time.time() - start

            logger.info(
                f"{name} took {ellapsed:.2f}s"
            )

            return result
        
        return wrapper
    return decorator

