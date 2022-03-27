from torch import nn
from torch import functional as F
import numpy as np
from data_loader import (MITBIHDataLoader, PTBDataLoader, DataLoaderUtil,
                            DATA_MITBIH, DATA_PTBDB, DataLoaderUtilMini)
import torch
from torch.optim.adam import Adam

import copy
import logging
from sklearn.metrics import accuracy_score, f1_score
from util import get_timestamp_str

class GlobalMaxPooling(nn.Module):
    def __init__(self, dims=None) -> None:
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        x, _ = torch.max(x, dim=self.dims, keepdim=True) # assuming dim 1 is channel dim
        return x

class CnnModel2DMITBIH(nn.Module):
    def __init__(self, config={"num_classes": 5}) -> None:
        super().__init__()
        self.pad_size_before_reshape = 0
        self.input_feature_count = 187
        self.height = 17
        assert (self.input_feature_count+self.pad_size_before_reshape)%\
            self.height == 0
        self.width = (self.input_feature_count+self.pad_size_before_reshape)//\
            self.height
        self.num_classes = config["num_classes"]
        
        self._build_network()

    def _build_network(self):
        
        self.block_0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.classification_head = self._create_classification_head()
    
    def _create_classification_head(self):
        """
        Create classification head which takes feature
        extracted by the CNN
        """
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out_ = torch.reshape(x, (x.shape[0], 1, self.height, self.width))
        out_ = self.block_0(out_)
        out_ = self.block_1(out_)
        out_ = self.block_2(out_)
        out_, _ = torch.max(out_, dim=2, keepdim=True)

        out_ = self.classification_head(out_)

        return out_

