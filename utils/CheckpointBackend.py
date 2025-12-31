import pdb, os, logging
from abc import ABC, abstractmethod
from pathlib import Path
import torch
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError


class CheckpointBackendBase():
    def __init__(self):
        self.logger = logging.getLogger("Backend")

    @abstractmethod
    def save(self, state: dict, path: str):
        pass
    
    @abstractmethod
    def load(self, path: str) -> dict:
        pass

class LocalCheckpointBackend(CheckpointBackendBase):
    def __init__(self, config):
        super().__init__()
        self.path = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_file)

    def save(self, state: dict):
        local_save(state, self.path)

    def load(self, device='cpu') -> dict:
        return torch.load(self.path, map_location=device)

class S3CheckpointBackend(CheckpointBackendBase):
    def __init__(self, config):
        super().__init__()
        self.path = os.path.join(config.checkpoint_dir, config.checkpoint_file)
        self.bucket = config.s3bucket

    def save(self, state: dict):
        try:
            local_save(state, self.path)
            self.logger.info("Successfully saved checkpoint locally...")
        except Exception as e:
            self.logger.info("Failed to save locally")
        self.upload_file() 


    def load(self, path: str):
        return local_load(path)

    def upload_file(self):
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param bucket: Bucket to upload to
        :param object_name: S3 object name. If not specified then file_name is used
        :return: True if file was uploaded, else False
        """
        try:
            file_name = self.path
            bucket = self.bucket
            s3 = boto3.client('s3')
            with open(file_name, "rb") as f:
                s3.upload_fileobj(f, bucket, file_name)
        except ClientError as e:
            self.logger.info("AWS ClientError, failed to upload to S3")
        except NoCredentialsError as e:
            self.logger.info("AWS Credentials Error, failed to upload to S3")
        except PartialCredentialsError as e:
            self.logger.info("AWS Credentials Error, failed to upload to S3")



def build_checkpoint_backend(config):
    backend_type = config.checkpoint_backend
    if backend_type == 's3':
        return S3CheckpointBackend(config)
    elif backend_type == 'local':
        return LocalCheckpointBackend(config)
    else:
        raise ValueError("Unknown checkpoint backend type")

def local_save(state: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def local_load(path: str, device='cpu') -> dict:
    return torch.load(path, map_location=device)
    
    
