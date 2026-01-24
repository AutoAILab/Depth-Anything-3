# Utility Functions Guidelines

## File and Path Handling
- Use `pathlib.Path` for all path operations
- Implement robust path validation
- Handle both relative and absolute paths
- Use `os.path` only when necessary for compatibility

## Data Processing
- Use NumPy for numerical computations
- Implement efficient array operations
- Handle different data formats (images, point clouds, etc.)
- Validate data integrity and types

## Configuration Management
- Use YAML or JSON for configuration files
- Implement default values and validation
- Support environment variable overrides
- Document all configuration options

## Logging
- Use Python's `logging` module
- Implement different log levels appropriately
- Add timestamps and module information
- Support log file output and rotation

## Memory Management
- Monitor memory usage for large datasets
- Implement garbage collection hints
- Use context managers for resource cleanup
- Handle GPU memory efficiently

## Parallel Processing
- Use `concurrent.futures` for CPU parallelism
- Implement proper thread/process safety
- Use `torch.multiprocessing` for PyTorch operations
- Handle inter-process communication

## Error Handling
- Create custom exception classes
- Provide detailed error messages
- Implement retry mechanisms for transient failures
- Log errors with context information

## Testing Utilities
- Create fixtures for common test data
- Implement mock objects for external dependencies
- Use `pytest` for testing framework
- Write comprehensive unit and integration tests