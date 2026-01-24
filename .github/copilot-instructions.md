# GitHub Copilot Instructions for Depth Anything 3

This file contains project-specific instructions for GitHub Copilot to help with coding in this repository.

## Project Overview
Depth Anything 3 is a state-of-the-art depth estimation model that can predict depth from single images or videos. This repository contains the implementation, CLI tools, and web application.

## Coding Guidelines
- Use Python 3.8+ features
- Follow PEP 8 style guidelines
- Use type hints where possible
- Prefer functional programming where appropriate
- Use Typer for CLI commands
- Use Gradio for web interfaces

## Key Libraries
- PyTorch for deep learning
- NumPy for numerical computations
- OpenCV for image processing
- Typer for CLI
- Gradio for web UI

## Project Structure
- `src/depth_anything_3/`: Main package
- `scripts/`: Utility scripts
- `docs/`: Documentation
- `assets/`: Example data and outputs

## Common Patterns
- Use the `typer` library for command-line interfaces
- Handle file I/O with proper validation
- Use logging for debugging and monitoring
- Implement proper error handling

## Task Management Workflow

For new development tasks, follow this structured workflow using the `.tickets/` directory:

1. **Task Initiation**: Create a markdown file in `.tickets/` for each task (e.g., `.tickets/feature-xyz.md`)

2. **User Summary**: User provides an initial summary of the work to be done

3. **Clarification & Review**: AI asks for clarification and provides initial review of the requirements

4. **AI Summary**: AI writes a comprehensive summary of the work, including scope, requirements, and approach

5. **User Review**: User reviews and approves the AI summary

6. **Checklist Creation**: AI adds a detailed checklist of work items to the ticket file

7. **Implementation**: Once user instructs to proceed, AI updates the checklist with progress as work is completed

8. **Completion**: When a ticket is complete, move the ticket task file to `.complete/`

**Note**: When working on tickets, if the work requires changes or updates to the project design, always keep the design docs in `docs/design/overview.md` up to date.

## Specialized Instructions
For detailed guidelines on specific components, see the sub-instruction files:
- [Python Development](copilot-instructions/python.md)
- [PyTorch Model Development](copilot-instructions/pytorch.md)
- [CLI with Typer](copilot-instructions/cli.md)
- [Gradio Web Apps](copilot-instructions/gradio.md)
- [Utility Functions](copilot-instructions/utils.md)
- [API Development](copilot-instructions/api.md)

## Notes
- GPU acceleration is preferred for inference
- Model weights are loaded from Hugging Face or local directories
- Export formats include GLB, NPZ, and feature visualizations