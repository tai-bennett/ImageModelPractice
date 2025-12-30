import torch, logging, pdb
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class FashionMNISTDataLoader():
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("FraudDataLoader")
        self.logger.info("Loading DATA...")
        transform = transforms.ToTensor()
        self.dataset = torchvision.datasets.FashionMNIST(config.data_path, train=True, download=True, transform=transform)

        self.train_loader = DataLoader(self.dataset, batch_size=self.config.batch_size)
