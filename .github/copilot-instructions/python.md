# Python Coding Guidelines

## General Python Practices
- Use Python 3.8+ features and syntax
- Follow PEP 8 style guidelines strictly
- Use type hints for all function parameters and return values
- Prefer descriptive variable and function names
- Use docstrings for all public functions and classes
- Keep functions small and focused on single responsibilities

## Imports
- Use absolute imports within the package
- Group imports: standard library, third-party, local
- Use `from __future__ import annotations` for forward references

## Error Handling
- Use specific exception types
- Provide meaningful error messages
- Use context managers where appropriate

## Performance
- Use list comprehensions and generator expressions
- Avoid unnecessary computations in loops
- Use appropriate data structures (dicts, sets, etc.)

## Code Organization
- Separate concerns into different modules
- Use classes for related functionality
- Keep module-level code minimal