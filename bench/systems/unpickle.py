"""Load a model pickled by the ReAct sandbox, allowing only known-safe classes.

A pickle can execute arbitrary code when loaded. The ReAct agent's final model was written
by model-generated code, so it is loaded with a restricted unpickler that resolves only
classes from scikit-learn, LightGBM, NumPy, SciPy and pandas (plus plain builtin types).
Anything else (``os.system``, ``builtins.eval``, ``getattr`` chains, ...) is refused before
it can run. Tests load real scikit-learn and LightGBM pipelines through it and check that
crafted malicious pickles are refused.
"""

import io
import pickle
from typing import Any, Final

ALLOWED_MODULE_ROOTS: Final = frozenset(
    {"sklearn", "lightgbm", "numpy", "scipy", "pandas", "copyreg", "collections", "_codecs"}
)
#: Builtins a scikit-learn / numpy pickle legitimately references.
ALLOWED_BUILTINS: Final = frozenset(
    {
        "object", "list", "dict", "tuple", "set", "frozenset", "slice", "range", "complex",
        "float", "int", "bool", "str", "bytes", "bytearray",
    }
)  # fmt: skip
#: Never allowed, even under an allowed root (they reach the system or load code).
BLOCKED_SUFFIXES: Final = ("system", "popen", "exec", "eval", "load", "loads", "__import__")


class UnsafePickleError(pickle.UnpicklingError):
    """The pickle references a class or function outside the allowlist."""


#: Cython classes scikit-learn pickles under a bare module name (HistGradientBoosting's
#: loss objects): exactly these aliases are resolved to their real, allowed module.
MODULE_ALIASES: Final = {"_loss": "sklearn._loss._loss"}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        module = MODULE_ALIASES.get(module, module)
        parts = module.split(".")
        builtin_ok = module == "builtins" and name in ALLOWED_BUILTINS
        library_ok = (
            parts[0] in ALLOWED_MODULE_ROOTS
            and not name.lower().endswith(BLOCKED_SUFFIXES)
            and not any(part in {"os", "subprocess", "sys", "builtins"} for part in parts)
        )
        if builtin_ok or library_ok:
            return super().find_class(module, name)
        raise UnsafePickleError(f"refusing to load {module}.{name} from a model pickle")


def restricted_loads(data: bytes) -> Any:
    return RestrictedUnpickler(io.BytesIO(data)).load()
