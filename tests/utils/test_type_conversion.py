"""Tests for type_conversion module."""

import numpy as np
import pytest

from src.utils.type_conversion import (
  is_numpy_scalar,
  numpy_to_python_scalar,
  safe_float,
  safe_int,
)


class TestNumpyToPythonScalar:
  """Test numpy_to_python_scalar function."""

  def test_numpy_int_to_python_float(self):
    """Test converting numpy int to Python float."""
    numpy_int = np.int32(42)
    result = numpy_to_python_scalar(numpy_int)
    assert isinstance(result, float)
    assert result == 42.0

  def test_numpy_float_to_python_float(self):
    """Test converting numpy float to Python float."""
    numpy_float = np.float64(3.14)
    result = numpy_to_python_scalar(numpy_float)
    assert isinstance(result, float)
    assert result == 3.14

  def test_python_int_to_python_float(self):
    """Test converting Python int to Python float."""
    python_int = 42
    result = numpy_to_python_scalar(python_int)
    assert isinstance(result, float)
    assert result == 42.0

  def test_python_float_unchanged(self):
    """Test that Python float is returned as float."""
    python_float = 3.14
    result = numpy_to_python_scalar(python_float)
    assert isinstance(result, float)
    assert result == 3.14


class TestSafeFloat:
  """Test safe_float function."""

  def test_numpy_int_to_float(self):
    """Test converting numpy int to Python float."""
    numpy_int = np.int64(100)
    result = safe_float(numpy_int)
    assert isinstance(result, float)
    assert result == 100.0

  def test_numpy_float_to_float(self):
    """Test converting numpy float to Python float."""
    numpy_float = np.float32(2.5)
    result = safe_float(numpy_float)
    assert isinstance(result, float)
    assert result == 2.5

  def test_python_int_to_float(self):
    """Test converting Python int to Python float."""
    python_int = 42
    result = safe_float(python_int)
    assert isinstance(result, float)
    assert result == 42.0

  def test_python_float_unchanged(self):
    """Test that Python float is returned unchanged."""
    python_float = 3.14159
    result = safe_float(python_float)
    assert isinstance(result, float)
    assert result == 3.14159


class TestSafeInt:
  """Test safe_int function."""

  def test_numpy_int_to_int(self):
    """Test converting numpy int to Python int."""
    numpy_int = np.int32(42)
    result = safe_int(numpy_int)
    assert isinstance(result, int)
    assert result == 42

  def test_numpy_float_to_int(self):
    """Test converting numpy float to Python int."""
    numpy_float = np.float64(3.7)
    result = safe_int(numpy_float)
    assert isinstance(result, int)
    assert result == 3  # Should truncate

  def test_python_float_to_int(self):
    """Test converting Python float to Python int."""
    python_float = 2.9
    result = safe_int(python_float)
    assert isinstance(result, int)
    assert result == 2  # Should truncate

  def test_python_int_unchanged(self):
    """Test that Python int is returned unchanged."""
    python_int = 42
    result = safe_int(python_int)
    assert isinstance(result, int)
    assert result == 42


class TestIsNumpyScalar:
  """Test is_numpy_scalar function."""

  def test_numpy_int_is_scalar(self):
    """Test that numpy int is recognized as scalar."""
    numpy_int = np.int16(42)
    assert is_numpy_scalar(numpy_int) is True

  def test_numpy_float_is_scalar(self):
    """Test that numpy float is recognized as scalar."""
    numpy_float = np.float32(3.14)
    assert is_numpy_scalar(numpy_float) is True

  def test_numpy_bool_is_scalar(self):
    """Test that numpy bool is recognized as scalar."""
    numpy_bool = np.bool_(True)
    assert is_numpy_scalar(numpy_bool) is True

  def test_python_int_not_scalar(self):
    """Test that Python int is not recognized as numpy scalar."""
    python_int = 42
    assert is_numpy_scalar(python_int) is False

  def test_python_float_not_scalar(self):
    """Test that Python float is not recognized as numpy scalar."""
    python_float = 3.14
    assert is_numpy_scalar(python_float) is False

  def test_python_bool_not_scalar(self):
    """Test that Python bool is not recognized as numpy scalar."""
    python_bool = True
    assert is_numpy_scalar(python_bool) is False

  def test_numpy_array_not_scalar(self):
    """Test that numpy array is not recognized as scalar."""
    numpy_array = np.array([1, 2, 3])
    assert is_numpy_scalar(numpy_array) is False

  def test_string_not_scalar(self):
    """Test that string is not recognized as numpy scalar."""
    string_value = "test"
    assert is_numpy_scalar(string_value) is False

  def test_none_not_scalar(self):
    """Test that None is not recognized as numpy scalar."""
    none_value = None
    assert is_numpy_scalar(none_value) is False


class TestTypeConversionEdgeCases:
  """Test edge cases for type conversion functions."""

  def test_safe_float_with_inf(self):
    """Test safe_float with infinity values."""
    inf_value = float("inf")
    result = safe_float(inf_value)
    assert result == float("inf")

  def test_safe_float_with_nan(self):
    """Test safe_float with NaN values."""
    nan_value = float("nan")
    result = safe_float(nan_value)
    assert str(result) == "nan"  # NaN != NaN, so check string representation

  def test_safe_int_with_inf(self):
    """Test safe_int with infinity (should raise OverflowError)."""
    inf_value = float("inf")
    with pytest.raises(OverflowError):
      safe_int(inf_value)

  def test_safe_int_with_nan(self):
    """Test safe_int with NaN (should raise ValueError)."""
    nan_value = float("nan")
    with pytest.raises(ValueError):
      safe_int(nan_value)

  def test_numpy_to_python_scalar_with_complex(self):
    """Test numpy_to_python_scalar with complex numbers."""
    complex_value = np.complex64(1 + 2j)
    with pytest.raises(TypeError, match="Cannot convert complex number to float"):
      numpy_to_python_scalar(complex_value)

  def test_large_numpy_int(self):
    """Test conversion of large numpy integers."""
    large_int = np.int64(9223372036854775807)  # Max int64
    result = safe_int(large_int)
    assert isinstance(result, int)
    assert result == 9223372036854775807

  def test_numpy_float_precision(self):
    """Test that numpy float precision is preserved."""
    precise_float = np.float64(1.23456789012345)
    result = safe_float(precise_float)
    assert isinstance(result, float)
    assert result == 1.23456789012345
