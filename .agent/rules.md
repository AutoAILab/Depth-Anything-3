# Depth Anything 3 Workspace Rules

## Project Overview
Depth Anything 3 is a state-of-the-art depth estimation model. This repository contains the implementation, CLI tools, and web application.

## Project Structure
- `src/depth_anything_3/`: Main package
- `scripts/`: Utility scripts
- `docs/`: Documentation
- `assets/`: Example data and outputs
- `.tickets/`: Active task management (see Workflows)
- `.complete/`: Completed tasks

## Coding Guidelines

### General Python
- **Version**: Python 3.8+
- **Style**: Follow PEP 8 strictly.
- **Typing**: Use type hints for all parameters and return values.
- **Documentation**: Use docstrings for all public functions and classes.
- **Imports**: Use absolute imports within the package. Group as: standard library, third-party, local. Use `from __future__ import annotations`.

### CLI Development (Typer)
- Use `typer.Typer` for apps and subcommands.
- Use `pathlib.Path` for file paths.
- Provide `help` text for all parameters.
- Use `rich` or `tqdm` for progress bars.

### Web Application (Gradio / FastAPI)
- **Gradio**: Use `gr.Blocks` for complex layouts. Use appropriate components (`gr.Image`, `gr.Video`). Implement loading states.
- **FastAPI**: Use Pydantic for request/response models. Use async/await for I/O. Implement API key authentication.

### PyTorch Model Development
- Use `torch.nn.Module` for all components.
- Use `torch.no_grad()` for inference.
- Implement proper device handling (CPU/GPU).
- Save/load with `state_dict`.
- Monitor memory with `torch.cuda.memory_summary()`.

### Utility & Data
- Use **NumPy** for numerical computations.
- Use **OpenCV** for image processing.
- Use **Pathlib** for all path operations.
- Use `logging` module for debugging and monitoring.

## Task Management Workflow
Follow the structured workflow in `.agent/workflows/ticket.md` for all new development tasks using the `.tickets/` and `.complete/` directories.
- Always keep `docs/design/overview.md` up to date if design changes.
