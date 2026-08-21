"""Bindings for Murty's ranked assignment algorithm.

Build the extension once with ``tracker-eval-build gnnpmb``.
"""

try:
    from ._murty import Murty
except ImportError as exc:  # pragma: no cover - depends on local C++ build
    raise ImportError(
        "GNN-PMB requires the Murty extension. Run "
        "'tracker-eval-build gnnpmb' after installing CMake, Eigen3 and pybind11."
    ) from exc

__all__ = ["Murty"]
