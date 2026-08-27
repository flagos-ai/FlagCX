import importlib
import os


VENDOR_BACKEND_PACKAGES = (
    "torch_npu",
    "torch_mlu",
    "torch_musa",
    "torch_txda",
    "torch_gcu",
    "torch_ptpu",
)
VALID_TORCH_BACKENDS = ("vendor", "flagos")


def selected_torch_backend():
    backend = os.environ.get("FLAGCX_TORCH_BACKEND", "vendor")
    backend = backend.strip().lower() or "vendor"
    if backend not in VALID_TORCH_BACKENDS:
        raise RuntimeError(
            f"Invalid FLAGCX_TORCH_BACKEND={backend!r}. "
            f"Valid values: {', '.join(VALID_TORCH_BACKENDS)}"
        )
    return backend


def load_torch_device_backend():
    """Load exactly one PrivateUse1 provider before loading flagcx._C."""
    if selected_torch_backend() == "flagos":
        importlib.import_module("torch_fl")
        return "torch_fl"

    for package in VENDOR_BACKEND_PACKAGES:
        try:
            importlib.import_module(package)
            return package
        except Exception:
            continue
    return None
