from torch import nn
from torch import functional as F
import numpy as np
from data_loader import MITBIHDataLoader, PTBDataLoader
import copy

MODEL_CNN_RES = "CNN with Residual Blocks"

DATA_MITBIH = "Dataset 1"
DATA_PTBDB = "Dataset 2"


class ResidualBlock(nn.Module):
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
        `identity_projection` : "it is the layer to project the input to 
                                desired shape if needed (when the shape of 
                                input does not match with the output)" or 
                                a boolean value to indicate such a layer 
                                should be used.
        """
        super().__init__()
        assert shortcut_connection_flags[-1] == 1
        if isinstance(identity_projection, bool):
            if identity_projection:
                self.identity_projection = \
                    self._create_identity_projection_block(
                        in_channels=num_input_channels,
                        out_channels=layerwise_num_ouput_channels[0],
                        kernel_size=kernel_sizes[0],
                        stride=strides[0]
                    )
        else:
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
        self.relu_op = nn.ReLU()


    
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

            out_ = self.relu_op(out_) # adding relu here as all the block ends
                                # in batchnorm
            output_of_last_block = out_

        return out_

    def _create_identity_projection_block(self, in_channels, out_channels,
            kernel_size, stride):
        # creates a block with conv layer + batchnorm layer 
        block = nn.Sequential(nn.Conv1d(in_channels=in_channels,
         out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                                 nn.BatchNorm1d(out_channels))
        return block
        




class CnnWithResidualBlocks(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.default_config = {
            "num_input_channels": 1,

            
            "first_transformation": {
                # this is the number of channels of the output of the 
                # first conv and batchnrom operation. This output is 
                # further fed to the residual blocks,
                "num_output_channels": 16,
                "kernel_size": 5,
                "padding": 2,
                "stride": 1
            },


            # this config is to control the general attributes of 
            # the residual block
            "residual_block_config": {
                # the stride to use when downsampling a map
                "downsampling_strides": [2, 2, 2],
                "kernel_sizes": [3, 3, 3],

                # this is the number of conv blocks used in each of the 
                # residual blocks
                "num_conv_layers_in_blocks": [2, 2, 2],

                # it is a list of number of output channels for each of the
                # residual block (in order)
                "num_filters": [32, 64, 128],


            }
        }

        # if config is provided then use that otherwise default one
        self.config = self.default_config
        if config is not None:
            self.config = config

        self._build_model()

    def forward(self, x) :
        out_ = self.intial_transformation(x)
        for block in self.residual_blocks:
            out_ = block(out_)
        
        return out_

    def get_conv_bn_relu_block(self, in_channels, out_channels,
            kernel_size, stride, padding):
        block = nn.Sequential(nn.Conv1d(in_channels=in_channels,
         out_channels=out_channels, kernel_size=kernel_size, stride=stride,
         padding=padding),
                                 nn.BatchNorm1d(out_channels),
                                 nn.ReLU())
        return block
    
    def get_residual_blocks(self, config, num_input_channels):
        # Creates a module list of residual blocks
        # based on the configuration provided

        # parameters for residual blocks
        rb_config = config["residual_block_config"]
        rb_downsampling_stride = rb_config["downsampling_strides"]
        rb_num_conv_layers_per_block = rb_config["num_conv_layers_in_blocks"]
        rb_num_filters = rb_config["num_filters"]
        rb_kernel_sizes = rb_config["kernel_sizes"]


        num_residual_blocks = len(rb_num_filters)

        # list of input channel for each of the sub blocks
        input_channels = [num_input_channels] + rb_num_filters[0:-1]

        all_residual_blocks = []

        for idx in range(num_residual_blocks):
            k = rb_kernel_sizes[idx]
            num_conv_blocks = rb_num_conv_layers_per_block[idx]
            downsampling_stride = rb_downsampling_stride[idx]
            num_out_channels = rb_num_filters[idx]

            output_channels = [num_out_channels]*num_conv_blocks
            strides = [downsampling_stride]+[1]*(num_conv_blocks - 1)
            kernel_sizes = [k]*num_conv_blocks
            paddings = [0] + [int(k/2)]*(num_conv_blocks - 1)


            residual_block = ResidualBlock(
                input_channels[idx], output_channels, kernel_sizes, strides,
                paddings, shortcut_connection_flags, True)
            

            all_residual_blocks.append(residual_block)

        return nn.ModuleList(all_residual_blocks)
    
    def _build_model(self):
        config = self.config
        num_input_channels = config["num_input_channels"]

        # parameters for the first transformation
        ft_config = config["first_transformation"]
        ft_kernel_size = ft_config["kernel_size"]
        ft_padding = ft_config["padding"]
        ft_output_channels = ft_config["num_output_channels"]
        ft_stride = ft_config["stride"]

        
        # first few layers which process the input before feeding it
        # to the residual block
        self.intial_transformation = self.get_conv_bn_relu_block(
            in_channels=num_input_channels,
            out_channels=ft_output_channels,
            kernel_size=ft_kernel_size,
            stride=ft_stride,
            padding=ft_padding
        )

        self.residual_blocks = self.get_residual_block(self.config,
                                        num_input_channels=ft_output_channels)



    

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

    output_channels = [16, 16, 16, 16]
    kernel_sizes = [3 for i in range(len(output_channels))]
    paddings = [1 for i in range(len(output_channels))]
    strides = [2, 1, 1, 1]
    shortcut_connection_flags = [0, 1]*2

    res_block = ResidualBlock(1, output_channels, kernel_sizes, strides,
                     paddings, shortcut_connection_flags, True)

    print(res_block)