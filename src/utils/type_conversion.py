"""Type conversion utilities for safe numeric operations.

Provides utilities for converting between NumPy types and Python types,
ensuring type safety in numerical computations.
"""

from typing import Any

import numpy as np


def numpy_to_python_scalar(value: Any) -> float:
  """Convert numpy scalar to Python float safely.

  Args:
      value: Numeric value (Python or NumPy scalar)

  Returns:
      Python float value
  """
  if isinstance(value, np.generic):
    item = value.item()
    if isinstance(item, complex):
      raise TypeError("Cannot convert complex number to float")
    return float(item)
  if isinstance(value, complex):
    raise TypeError("Cannot convert complex number to float")
  return float(value)


def safe_float(value: float | np.floating | np.integer | int) -> Any:
  """Safely convert numeric types to Python float.

  Handles both Python and NumPy numeric types safely.

  Args:
      value: Numeric value (Python or NumPy)

  Returns:
      Python float value
  """
  if isinstance(value, (np.floating, np.integer)):
    return float(value.item())
  return float(value)


def safe_int(value: int | np.integer | float | np.floating) -> int:
  """Safely convert numeric types to Python int.

  Args:
      value: Numeric value to convert

  Returns:
      Python int value
  """
  if isinstance(value, np.integer):
    return int(value.item())
  return int(float(value))


def is_numpy_scalar(value) -> bool:
  """Check if value is a NumPy scalar type.

  Args:
      value: Value to check

  Returns:
      True if value is a NumPy scalar
  """
  return isinstance(value, np.generic)
