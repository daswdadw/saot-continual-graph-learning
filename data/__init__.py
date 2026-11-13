"""Data processing utilities for the SAOT reproduction project.

Import augmentation APIs from :mod:`data.augment`.  Keeping this package
initializer lightweight also allows ``python -m data.augment`` to run without
loading the module twice.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neighbor_sampling import LayerWiseNeighborLoader
    from .replay_buffer import ReplayBuffer

__all__ = ["LayerWiseNeighborLoader", "ReplayBuffer"]


def __getattr__(name: str):
    """Lazily expose ReplayBuffer without importing it for module execution."""

    if name == "ReplayBuffer":
        from .replay_buffer import ReplayBuffer

        return ReplayBuffer
    if name == "LayerWiseNeighborLoader":
        from .neighbor_sampling import LayerWiseNeighborLoader

        return LayerWiseNeighborLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
