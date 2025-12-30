from abc import ABC, abstractmethod
from pathlib import Path
import torch


class CheckpointBackendBase():
    def __init__(self):
        pass

    @abstractmethod
    def save(self, state: dict, path: str):
        pass
    
    @abstractmethod
    def load(self, path: str) -> dict:
        pass

class LocalCheckpointBackend(CheckpointBackendBase):
    def __init__(self):
        pass

    def save(self, state: dict, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load(self, path: str, device='cpu') -> dict:
        return torch.load(path, map_location=device)

#class S3CheckpointBackend(CheckpointBackendBase):
#    def __init__(self):
#        pass


def build_checkpoint_backend(backend_type):
    if backend_type == 's3':
        raise NotImplementedError
    elif backend_type == 'local':
        return LocalCheckpointBackend()
    else:
        raise ValueError("Unknown checkpoint backend type")
    
