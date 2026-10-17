import os
import platform


def is_wsl() -> bool:
    """Returns True if running inside WSL"""
    return "microsoft" in platform.uname().release.lower()


def configure_pytorch_allocator() -> None:
    """expandable_segments causes driver crashes in WSL2; keep False unless on native Linux."""
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        if is_wsl():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        else: # Enable on Linux to reduce memory fragmentation
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

configure_pytorch_allocator()
