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

class ChannelMaxPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        x, _ = torch.max(x, dim=1, keepdim=True) # assuming dim 1 is channel dim
        return x

class CollateAllFeaturesToChannel (nn.Module):
    """
    Flattens the features and collects then along channel dimension
    """
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.unsqueeze(x, dim=2) # (batch, collected_features(channel), 1)
        return x



class GlobalMaxPooling(nn.Module):
    def __init__(self, dims=None) -> None:
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        x, _ = torch.max(x, dim=self.dims, keepdim=True) # assuming dim 1 is channel dim
        return x
class CnnBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 1
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
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            GlobalMaxPooling(dims=(2)),
            nn.Dropout(p=0.1),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        out_ = self.layers(x)
        out_ = out_.squeeze()
        return out_

class CnnEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 1
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
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            GlobalMaxPooling(dims=(2))
            # nn.Dropout(p=0.1),
            # nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3, stride=1, padding='same'),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            # nn.ReLU()
        )

    def forward(self, x):
        out_ = self.layers(x)
        return out_

class CnnDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            CollateAllFeaturesToChannel(),
            nn.Conv1d(in_channels=224, out_channels=187, kernel_size=1),
            nn.ReLU()
        )

    
    def forward(self, encoder_ouput):
        decoder_output = self.layers(encoder_ouput)
        decoder_output = torch.permute(decoder_output, (0, 2, 1))
        return decoder_output
        
class CnnEncoderDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = CnnEncoder()
        self.decoder = CnnDecoder()
    
    def forward(self, x):
        out_ = self.encoder(x)
        out_ = self.decoder(out_)
        return out_


class CnnPretrainEncoderWithTrainableClassifierHead(nn.Module):
    def __init__(self, config={"num_classes": 5}) -> None:
        super().__init__()
        # This model can load weights of trained 
        # CnnEncoderDecoder (it will only use the encoder weights though)
        self.num_classes = config["num_classes"]
        self.encoder = CnnEncoder()
        self.classifier = nn.Sequential(
            nn.Flatten(), # flatten the feature map
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=self.num_classes),
            nn.ReLU()
        )
        self.clasification_layer_activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        output_ = self.encoder(x)
        output_ = self.classifier(output_)
        output_ = self.clasification_layer_activation(output_)
        return output_
    
    def load_state_dict(self, state_dict, strict=False):
        # Since decoder keys are not needed and classifier keys
        # will be missing in the model_state_dict, therefore, setting 
        # strict to `False` (as default value)
        super().load_state_dict(state_dict, strict=strict)

        # Also freezing the encoder layers:
        for name, param in self.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = True # FIXME
            # Also make sure these weights are not passed to the optimizer


class CnnPretrainEncoderWithTrainableClassifierHeadPartiallyFrozen(
    CnnPretrainEncoderWithTrainableClassifierHead):
    def __init__(self, config={ "num_classes": 5 }) -> None:
        super().__init__(config)
    
    def load_state_dict(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict)

        #unfreeze encoder layers with these ids
        layer_nums = [12, 14]

        for name, param in self.named_parameters():
            if name.startswith("encoder."):
                layer_num = int(name.split(".")[2])
                if layer_num in layer_nums:
                    param.requires_grad = True

class CnnPretrainEncoderWithTrainableClassifierHeadPTB(
    CnnPretrainEncoderWithTrainableClassifierHead
):
    def __init__(self, config={ "num_classes": 1 }) -> None:
        super().__init__(config)
        self.clasification_layer_activation = nn.Sigmoid()

def test_load_model_weights(model: nn.Module, weights_path):
    model_state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict=model_state_dict)
    for k in model_state_dict:
        print(k)

if __name__ == "__main__":
    pass