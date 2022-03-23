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



class UnetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block_0 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU()
        )

        self.pool_0 = nn.MaxPool1d(kernel_size=2, return_indices=True)

        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )

        self.pool_1 = nn.MaxPool1d(kernel_size=2, return_indices=True)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding='same'),
            # nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )




    
    def forward(self, x):

        map_0 = self.block_0(x)

        output_, pool_indices_0 = self.pool_0(map_0)

        map_1 = self.block_1(output_)

        output_, pool_indices_1 = self.pool_1(map_1)

        bottleneck_output = self.bottleneck(output_)

        return [bottleneck_output, map_1, map_0], [pool_indices_1, pool_indices_0]


class UnetDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.expand_bottleneck_channel = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )

        self.unpool_1 = nn.MaxUnpool1d(kernel_size=2)

        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU()
        )

        self.unpool_0 = nn.MaxUnpool1d(kernel_size=2)

        self.block_0 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU()
        )

        self.output_head = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )


    
    def forward(self, encoder_ouputs, pooling_indices):
        bottleneck_output, enc_1, enc_0 = encoder_ouputs
        bottleneck_output = self.expand_bottleneck_channel(bottleneck_output)
        indices_1, indices_0 = pooling_indices

        unpooled_1 = self.unpool_1(bottleneck_output, indices_1,
                                    output_size=enc_1.shape)

        m = torch.cat([unpooled_1, enc_1], dim=1) # concatenate along channel
        # dimension

        m = self.block_1(m)

        unpooled_0 = self.unpool_0(m, indices_0,
                                    output_size=enc_0.shape)

        m = torch.cat([unpooled_0, enc_0], dim=1)

        output_ = self.output_head(m)

        return output_
        
class UnetEncoderDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = UnetEncoder()
        self.decoder = UnetDecoder()
    
    def forward(self, x):
        output_, pooling_indices = self.encoder(x)
        output_ = self.decoder(output_, pooling_indices)
        return output_


class UnetPretrainEncoderWithTrainableClassifierHead(nn.Module):
    def __init__(self, config={"num_classes": 5}) -> None:
        super().__init__()
        self.num_classes = config["num_classes"]
        self.encoder = UnetEncoder()
        self.classifier = nn.Sequential(
            nn.Flatten(), # flatten the feature map
            nn.Linear(in_features=46, out_features=16), # 
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=self.num_classes),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        output_, pooling_indices = self.encoder(x)
        z = output_[0] # feature in low dimensional latent space
        output_ = self.classifier(z)
        output_ = self.softmax(output_)
        return output_
    
    def load_state_dict(self, state_dict, strict=False):
        # Since decoder keys are not needed and classifier keys
        # will be missing in the model_state_dict, therefore, setting 
        # strict to `False` (as default value)
        super().load_state_dict(state_dict, strict=strict)

        # Also freezing the encoder layers:
        for name, param in self.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False
            # Also make sure these weights are not passed to the optimizer

def test_load_model_weights(model: nn.Module, weights_path):
    model_state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict=model_state_dict)
    for k in model_state_dict:
        print(k)

if __name__ == "__main__":
    encoder = UnetEncoder()
    from data_loader import DataLoader

    dataloader_util = DataLoaderUtil()

    dataset_name = DATA_MITBIH  # DATA_PTBDB # or DATA_MITBIH
    model_class = UnetEncoder

    train_loader, val_loader, test_loader \
        = dataloader_util.get_data_loaders(dataset_name, train_batch_size=200, 
        val_batch_size=1, test_batch_size=100, train_shuffle=False,
        val_split=0.1)
    
    encoder = UnetEncoder()
    decoder = UnetDecoder()
    unet_enc_dec = UnetEncoderDecoder()
    unet_enc_classifier = UnetPretrainEncoderWithTrainableClassifierHead()

    weights_path = "runs/2022-03-23_132933__unet_ae/best_model.ckpt"

    test_load_model_weights(unet_enc_dec, weights_path)
    test_load_model_weights(unet_enc_classifier, weights_path)


    for batch_data in train_loader:
        x, y = batch_data
        z = encoder(x)
        t = decoder(*z)
        g = z