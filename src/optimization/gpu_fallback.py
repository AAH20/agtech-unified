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
        """Try GPU solver first, fall back to CPU on failure."""
        try:
            return gpu_solver(problem)
        except Exception as e:
            logger.warning(f"GPU solver failed: {e}. Falling back to CPU.")
            return cpu_solver(problem)
