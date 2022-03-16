from torch import nn
from torch import functional as F
import numpy as np
from data_loader import MITBIHDataLoader, PTBDataLoader

MODEL_CNN_RES = "CNN with Residual Blocks"

DATA_MITBIH = "Dataset 1"
DATA_PTBDB = "Dataset 2"


class ResidualBlockSmall(nn.Module):
    def __init__(self, num_input_channels, layerwise_num_ouput_channels,
                kernel_sizes, strides, paddings, shortcut_connection_flags,
                identity_projection = None ) -> None:
        """
        `num_input_channels`  : number of channels in the input
        `layerwise_num_ouput_channels`  : list of number of channels in the 
                                       output for each of the convolution
                                       layer
        `kernel_sizes` : list of kernel size for each of the layer
        `strides`  : list of stride to use for each of the layer
        `paddings` : list of padding value to use for each of the layer
        `shortcut_connection_flags`: list of 0s and 1s, 1 at a give index
                                     indicates a shortcut connection at the
                                     layer at that particular index should
                                     be added (Note: last entry must be 1)
        `identity_projection` : it is the layer to project the input to 
                                desired shape if needed (when the shape of 
                                input does not match with the output)
        """
        super().__init__()
        assert shortcut_connection_flags[-1] == 1
        self.identity_projection = identity_projection
        self.shortcut_connection_flags = shortcut_connection_flags

        # num of sets of conv-batchnorm-relu blocks to use
        num_block_units = len(layerwise_num_ouput_channels)

        layer_blocks = [] # list of layers grouped into sub blocks


        current_block = [] # running list of layers in current block

        for idx in range(num_block_units):
            if idx == 0:
                in_channels = num_input_channels
            else:
                # use num of channels in the last block's output
                in_channels = layerwise_num_ouput_channels[idx - 1]
            

            out_channels = layerwise_num_ouput_channels[idx]

            conv_layer = nn.Conv1d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_sizes[idx],
            stride=strides[idx], padding=paddings[idx]
            )

            batchnorm_layer = nn.BatchNorm1d(num_features=out_channels)

            current_block.append(conv_layer)
            current_block.append(batchnorm_layer)

            if shortcut_connection_flags[idx] == 1:
                sequential_group = nn.Sequential(*current_block)
                layer_blocks.append(sequential_group)
                # track new block
                current_block = []

            else:
                relu_layer = nn.ReLU()
                current_block.append(relu_layer)
        
        self.blocks = nn.ModuleList(layer_blocks)


    
    def forward(self, x):
        
        out_ = x
        output_of_last_block = x
        for idx, block in enumerate(self.blocks):
            out_ = block(out_)
            if idx == 0 and (self.identity_projection is not None) :
                # basically transform x to match the shape of the current
                # output so that they can be addded together
                output_of_last_block = self.identity_projection(
                    output_of_last_block)

            # Shortcut connection
            out_ = out_ + output_of_last_block

            out_ = F.relu(out_) # adding relu here as all the block ends
                                # in batchnorm
            output_of_last_block = out_

        return out_


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
    trainer.train()