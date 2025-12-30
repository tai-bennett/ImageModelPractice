import pdb
import torch
from torch import nn
from src.data.dataset import FashionMNISTDataLoader
from agents.base import BaseAgent
from tqdm import tqdm


class FashionNetAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.logger.info("Loading datasets...")
        self.data_loader = FashionMNISTDataLoader(config)
        self.logger.info("Building model...")

    def run(self):
        """
        The main operator
        :return:
        """
        self.logger.info("Running agent...")
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        pbar = tqdm(range(self.config.max_epoch))
        for epoch in pbar:
            self.train_one_epoch()
            #val_loss = self.validate()
            #pbar.set_postfix(val_loss=f"{val_loss:.4f}")

    def train_one_epoch(self):
        pass

    def validate(self):
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Finalizing...")
