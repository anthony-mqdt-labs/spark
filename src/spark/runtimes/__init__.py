"""Runtime backends. Importing this package registers the available adapters."""

from .base import Backend, entry_for_backend, get_backend, register, select_backend_for

# Import side effects register concrete backends.
from . import llama_cpp, mlx_lm, mlx_vlm, ollama, omlx, prism_ml, provider, relay  # noqa: F401,E402

__all__ = ["Backend", "entry_for_backend", "get_backend", "register", "select_backend_for"]
