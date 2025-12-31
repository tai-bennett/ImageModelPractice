import pdb, os
import torch
from torch import nn
from src.data.dataset import FashionMNISTDataLoader
from agents.base import BaseAgent
from graphs.models.fashionnet import FashionNet
from tqdm import tqdm
from utils.metrics import logit_accuracy
from utils.CheckpointBackend import build_checkpoint_backend
from utils.misc import get_device, print_cuda_statistics


class FashionNetAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # machine learning
        self.device = get_device()
        self.data_loader = FashionMNISTDataLoader(config)
        self.logger.info("Building model...")
        self.model = FashionNet(config)
        self.logger.info("Define loss...")
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=config.learning_rate)

        # backend
        self.logger.info("Checkpoints will be saved via " + str(config.checkpoint_backend))
        self.checkpointbackend = build_checkpoint_backend(config)

        # device
        self.device = get_device()
        if self.device.type == 'cuda':
            #torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        elif device.type == 'cpu':
            self.logger.info("Program will run on *****CPU*****")
        # checkpoints
        self.load_checkpoint()

    def save_checkpoint(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        #path = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_file)
        self.checkpointbackend.save(state)
            

    def load_checkpoint(self):
        path = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_file)
        try:
            self.logger.info("Loading checkpoint '{}'".format(path))
            checkpoint = self.checkpointbackend.load()

            # self.current_epoch = checkpoint['epoch']
            # self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
            #      .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
            self.logger.info("Checkpoint loaded successfully")
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

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
            val_loss, val_acc = self.validate()
            pbar.set_postfix(val_acc=f"{val_acc:.4f}")

    def train_one_epoch(self):
        train_loss = 0
        for batch, (X, y) in enumerate(self.data_loader.train_loader):
            self.model.train()
            X.to(self.device, non_blocking=True)
            y.to(self.device, non_blocking=True)
            y_pred = self.model(X)
            loss = self.loss(y_pred, y)
            train_loss += loss
            if torch.isnan(loss):
                pdb.set_trace()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self):
        test_loss, test_acc = 0, 0
        self.model.eval()
        with torch.inference_mode():
            for X, y in self.data_loader.test_loader:
                X.to(self.device, non_blocking=True)
                y.to(self.device, non_blocking=True)
                y_pred = self.model(X)
                loss = self.loss(y_pred, y)
                test_loss += loss
                test_acc += logit_accuracy(y_pred, y, class_dim=1)
            test_loss = test_loss/len(self.data_loader.test_loader)
            test_acc = test_acc/len(self.data_loader.test_loader)
        return test_loss, test_acc

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Finalizing...")
        self.save_checkpoint()
