# ADR 0005: PyTorch for GPU Optimization

## Status

Accepted

## Context

The optimization module needs GPU acceleration for TSP/VRP solvers. We need to choose a GPU computing framework.

## Decision

We will use **PyTorch** for GPU-accelerated optimization.

### Rationale

1. **Tensor operations**: PyTorch's tensor operations are well-suited for distance matrix computations
2. **GPU acceleration**: Easy CPU/GPU code switching with `.to(device)`
3. **Ecosystem**: Large ecosystem for ML/AI integration
4. **Fallback**: CPU fallback when CUDA is not available

### Consequences

- **Positive**: GPU acceleration, familiar API, good documentation
- **Negative**: Large dependency (~2GB), requires CUDA for GPU support
- **Mitigation**: CPU fallback, optional dependency

## Alternatives Considered

1. **CuPy**: NumPy-like API but less flexible for custom algorithms
2. **TensorFlow**: Heavier, more complex API
3. **Numba**: JIT compilation but less control over GPU memory

## References

- `src/optimization/gpu_tsp.py` — GPUTSPSolver implementation
- `src/optimization/gpu_vrp.py` — GPUVRPSolver implementation
- `src/optimization/gpu_fallback.py` — GPUFallback implementation
- `tests/test_gpu_tsp.py` — GPU TSP tests
