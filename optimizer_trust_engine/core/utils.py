"""
Utility Functions for Optimizer Trust Engine
============================================

Production-grade utility functions with proper error handling.
"""

import hashlib
import json
import logging
import time
import random
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
from functools import wraps
from contextlib import contextmanager
import threading
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


class NumpyFallback:
    """Fallback implementation for numpy functions when numpy is not available"""
    
    @staticmethod
    def mean(values: Union[List[float], Dict[str, float]]) -> float:
        """Calculate mean of values"""
        if isinstance(values, dict):
            values = list(values.values())
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    @staticmethod
    def std(values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    @staticmethod
    def median(values: List[float]) -> float:
        """Calculate median of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        return sorted_values[n//2]
    
    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


# Try to import numpy, fall back if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = NumpyFallback()
    HAS_NUMPY = False


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


def validate_input(schema: Dict[str, Any]) -> Callable:
    """
    Decorator for validating function inputs against a schema
    
    Args:
        schema: Dictionary defining parameter validation rules
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Validate kwargs against schema
            for param, rules in schema.items():
                if param in kwargs:
                    value = kwargs[param]
                    
                    # Check type
                    if 'type' in rules and not isinstance(value, rules['type']):
                        raise TypeError(f"Parameter '{param}' must be of type {rules['type'].__name__}")
                    
                    # Check range
                    if 'min' in rules and value < rules['min']:
                        raise ValueError(f"Parameter '{param}' must be >= {rules['min']}")
                    if 'max' in rules and value > rules['max']:
                        raise ValueError(f"Parameter '{param}' must be <= {rules['max']}")
                    
                    # Check allowed values
                    if 'choices' in rules and value not in rules['choices']:
                        raise ValueError(f"Parameter '{param}' must be one of {rules['choices']}")
                
                # Check required parameters
                elif rules.get('required', False):
                    raise ValueError(f"Required parameter '{param}' is missing")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{name} took {elapsed:.3f} seconds")


def generate_unique_id(prefix: str = "", length: int = 16) -> str:
    """
    Generate a unique identifier
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part (in hex characters)
    
    Returns:
        Unique identifier string
    """
    timestamp = str(int(time.time() * 1000000))
    random_part = hashlib.sha256(
        f"{timestamp}{random.random()}".encode()
    ).hexdigest()[:length]
    
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def calculate_hash(data: Union[str, Dict, List]) -> str:
    """
    Calculate SHA256 hash of data
    
    Args:
        data: Data to hash (string, dict, or list)
    
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True)
    elif not isinstance(data, str):
        data = str(data)
    
    return hashlib.sha256(data.encode()).hexdigest()


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max"""
    return max(min_val, min(value, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values"""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0:
        return default
    return numerator / denominator


class ThreadSafeCounter:
    """Thread-safe counter implementation"""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value"""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value"""
        with self._lock:
            self._value -= amount
            return self._value
    
    @property
    def value(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value


class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, max_calls: int, period: float):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator implementation"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with self.lock:
                now = time.time()
                # Remove old calls outside the period
                self.calls = [call_time for call_time in self.calls 
                             if now - call_time < self.period]
                
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    # Retry after sleeping
                    return wrapper(*args, **kwargs)
                
                self.calls.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename[:255]  # Limit length