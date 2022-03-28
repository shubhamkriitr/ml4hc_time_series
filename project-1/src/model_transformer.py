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


class TransformerModelMITBIH(nn.Module):
    def __init__(self, config={"num_classes": 5}) -> None:
        super().__init__()
        self.num_classes = config["num_classes"]

        self.input_features_count = 1 # pass id as `d_model` to
        # TransformerEncoderLayer


        self.sequence_length = 187
        self.dim_feedforward = 128
        self.dropout = 0.1
        self.transformer_activation = "relu" # "relu", "gelu" or callabale
        # with one argument

        self.last_layer_activation = nn.Softmax(dim=1)
        
        self._build_network()

    def _build_network(self):
        self.encoder_layer_0 = nn.TransformerEncoderLayer(
            d_model=1,
            nhead=1,
            dim_feedforward=self.dim_feedforward,
            activation=self.transformer_activation,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model=1,
            nhead=1,
            dim_feedforward=self.dim_feedforward,
            activation=self.transformer_activation,
            dropout=self.dropout,
            batch_first=True
        )

        self.encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model=1,
            nhead=1,
            dim_feedforward=self.dim_feedforward,
            activation=self.transformer_activation,
            dropout=self.dropout,
            batch_first=True
        )

        # self.encoder_layer_3 = nn.TransformerEncoderLayer(
        #     d_model=1,
        #     nhead=1,
        #     dim_feedforward=self.dim_feedforward,
        #     activation=self.transformer_activation,
        #     dropout=self.dropout,
        #     batch_first=True
        # )

        self.classification_head = self._create_classification_head()

    def _create_classification_head(self):
        return nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=93, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=32, out_features=self.num_classes),
        )

    def forward(self, x):
        out_ = torch.permute(x, (0, 2, 1))

        # input dimensions should correspond to (batch, sequence, emdedding)
        out_ = self.encoder_layer_0(out_)
        out_ = self.encoder_layer_1(out_)
        out_ = self.encoder_layer_2(out_)
        # out_ = self.encoder_layer_3(out_)


        out_ = torch.permute(out_, (0, 2, 1))

        out_ = self.classification_head(out_)
        
        return out_
    
    def predict(self, x):
        out_ = self.forward(x)
        out_ = self.last_layer_activation(out_)
        return out_


class TransformerModelPTB(TransformerModelMITBIH):
    def __init__(self, config={ "num_classes": 5 }) -> None:
        super().__init__(config)
        self.last_layer_activation = nn.Sigmoid()
    
    def forward(self, x):
        out_ =  super().forward(x)
        out_ = self.last_layer_activation(out_)
        out_ = out_.squeeze()
        return out_
    
    def predict(self, x):
        return self.forward(x)
    
