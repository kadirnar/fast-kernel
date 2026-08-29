"""fast-kernel: autoresearch for model inference.

An agent harness that profiles a model, ranks its slow parts (Amdahl's law), and runs an
endless keep/revert optimization loop of hand-written CUDA C++ kernels captured with CUDA
graphs -- while every experiment is streamed to a live graph.

The package core is stdlib-only. Torch and the model libraries are imported lazily inside
the harness so that `fast-kernel status`, `dashboard` and `report` work anywhere.
"""

__version__ = "0.1.0"
