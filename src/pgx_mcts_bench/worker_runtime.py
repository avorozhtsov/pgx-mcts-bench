"""Shared process-worker setup for braid experiment runners."""

from __future__ import annotations

import os
from pathlib import Path


def enable_jax_compilation_cache() -> None:
    import jax

    cache = Path.home() / ".cache" / "jax-pgx-mcts-bench"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)


def worker_init() -> None:
    """Use process-level parallelism without nested BLAS thread pools."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    # PyTorch permits setting the inter-op pool only once, before parallel work
    # starts.  Sequential experiment runners legitimately call ``worker_init``
    # once per job in the same process, so avoid repeating the one-shot setter.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    enable_jax_compilation_cache()
