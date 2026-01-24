# PyTorch Development Guidelines

## Model Architecture
- Use `torch.nn.Module` for all neural network components
- Implement `forward` method clearly
- Use appropriate activation functions and normalization layers
- Document model input/output shapes and data types

## Training and Inference
- Use `torch.no_grad()` for inference
- Implement proper device handling (CPU/GPU)
- Use mixed precision training when possible
- Handle batch processing efficiently

## Data Handling
- Use `torch.utils.data.Dataset` and `DataLoader`
- Implement proper data augmentation
- Handle memory efficiently with pinned memory for GPU

## Best Practices
- Use `torch.jit.script` for performance-critical code
- Implement gradient clipping to prevent exploding gradients
- Use `torch.nn.utils.clip_grad_norm_`
- Save/load models with state_dict

## Debugging
- Use `torch.autograd.detect_anomaly()` for debugging gradients
- Monitor memory usage with `torch.cuda.memory_summary()`
- Use tensorboard or wandb for logging

## Performance Optimization
- Use `torch.compile` for PyTorch 2.0+
- Implement custom CUDA kernels when necessary
- Profile with `torch.profiler`