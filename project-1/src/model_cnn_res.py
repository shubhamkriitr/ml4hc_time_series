from tkinter.messagebox import NO
from torch import nn
from torch import functional as F
import numpy as np
from data_loader import MITBIHDataLoader, PTBDataLoader

MODEL_CNN_RES = "CNN with Residual Blocks"

DATA_MITBIH = "Dataset 1"
DATA_PTBDB = "Dataset 2"

class CnnWithResidualBlocks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    

    def forward(self, *input, **kwargs) :
        return super().forward(*input, **kwargs)
    

class BaseTrainer(object):

    def __init__(self, config) -> None:
        self.config = config
        self.dataloader = None

    def train(self):
        pass

    def prepare_data_loaders(self):
        if self.config["data"] == DATA_MITBIH:
            dataloader = MITBIHDataLoader()
        elif self.config["data"] == DATA_PTBDB:
            dataloader = PTBDataLoader()

        self.dataloader = dataloader
        
    def load_data(self):
        if self.dataloader is None:
            self.prepare_data_loaders()
        
        x_train, y_train, x_test, y_test = self.dataloader.load_data()

        return x_train, y_train, x_test, y_test



if __name__ == "__main__":
    trainer = BaseTrainer({"model": MODEL_CNN_RES, "data": DATA_MITBIH})

    trainer.load_data()