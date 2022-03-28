from torch import nn
from torch import functional as F
import numpy as np

import torch

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

class VanillaCnnPTB(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.classification_activation_layer = nn.Sigmoid()
        self._build_network()
    
    def _build_network(self):
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.num_classes)
            
        )

    def forward(self, x):
        out_ = self.layers(x)
        out_ = self.classification_activation_layer(out_)
        out_ = out_.squeeze()
        return out_
    
    def predict(self, x):
        return self.forward(x)

class VanillaCnnMITBIH(VanillaCnnPTB):
    def __init__(self, num_classes=5) -> None:
        super().__init__(num_classes)
        self.classification_activation_layer = nn.Softmax(dim=1)

    def forward(self, x):
        out_ = self.layers(x)
        return out_ # return just logits
    
    def predict(self, x):
        logits = self.forward(x)
        return self.classification_activation_layer(logits)


