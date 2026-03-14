import torch


def get_gpu_count() -> int:
    """Get number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0
