"""GPU fallback to CPU on CUDA failure."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class GPUFallback:
    """Tries GPU solver first, falls back to CPU on CUDA failure."""

    def __init__(self, device: Optional[str] = None):
        self.requested_device = device

    def solve(self, problem: Any, gpu_solver: Callable, cpu_solver: Callable) -> Any:
        """Try GPU solver first, fall back to CPU on CUDA failure."""
        if not callable(gpu_solver):
            raise TypeError("gpu_solver must be callable")
        if not callable(cpu_solver):
            raise TypeError("cpu_solver must be callable")

        try:
            return gpu_solver(problem)
        except (RuntimeError, OSError, ImportError) as e:
            if not self._is_cuda_error(e):
                raise
            logger.warning("GPU solver failed: %s. Falling back to CPU.", e, exc_info=True)
            try:
                return cpu_solver(problem)
            except Exception as cpu_e:
                raise RuntimeError(
                    f"Both GPU and CPU solvers failed. GPU error: {e}. CPU error: {cpu_e}"
                ) from cpu_e

    @staticmethod
    def _is_cuda_error(exc: Exception) -> bool:
        """Check if exception is CUDA-related."""
        # torch.cuda.OutOfMemoryError is a subclass of RuntimeError
        if "OutOfMemoryError" in type(exc).__name__:
            return True
        # Check for CUDA in the exception message
        if "CUDA" in str(exc) or "cuda" in str(exc):
            return True
        return False
