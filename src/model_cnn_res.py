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

logger = logging.getLogger(name=__name__)


class CnnWithResidualConnection(nn.Module):

    def __init__(self, config={"num_classes": 5}, *args, **kwargs) -> None:
        super().__init__()
        self.num_classes = config["num_classes"]

        self.expand_channel = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU())
        
        self.block_0 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU()
        )


        self.downsample_0 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )

        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )

        self.downsample_1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=16*2, out_channels=32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*2),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv1d(in_channels=32*2, out_channels=32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32*2, out_channels=32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*2),
            nn.ReLU()
        )

        self.downsample_2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=16*4, out_channels=32*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*4),
            nn.ReLU()
        )

        self.block_3 = nn.Sequential(
            nn.Conv1d(in_channels=32*4, out_channels=32*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*4),
            nn.ReLU(),
            nn.Conv1d(in_channels=32*4, out_channels=32*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*4),
            nn.ReLU()
        )

        self.downsample_3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=16*8, out_channels=32*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*8),
            nn.ReLU()
        )

        self.block_4 = nn.Sequential(
            nn.Conv1d(in_channels=32*8, out_channels=32*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*8),
            nn.ReLU(),
            nn.Conv1d(in_channels=32*8, out_channels=32*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*8),
            nn.ReLU()
        )

        self.downsample_4 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=32*8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU()
        )

        
        self.flatten = nn.Flatten()

        self.fc_block = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.num_classes),
            nn.ReLU()
        )
        self.last_layer_activation = nn.Softmax(dim=1)

        #generic operations
        self.relu = nn.ReLU()

        self.initialize_parameters()
        

    def forward(self, x):
        output_ = self.expand_channel(x)
        identity_output_ = output_

        output_ = self.block_0(output_)
        output_ = identity_output_ + output_ # shortcut connection
        output_ = self.downsample_0(output_)
        identity_output_ = output_

        output_ = self.block_1(output_)
        output_ = identity_output_ + output_ # shortcut connection
        output_ = self.downsample_1(output_)
        identity_output_ = output_

        output_ = self.block_2(output_)
        output_ = identity_output_ + output_ # shortcut connection
        output_ = self.downsample_2(output_)
        identity_output_ = output_

        output_ = self.block_3(output_)
        output_ = identity_output_ + output_ # shortcut connection
        output_ = self.downsample_3(output_)
        identity_output_ = output_

        output_ = self.block_4(output_)
        output_ = identity_output_ + output_ # shortcut connection
        
        output_ = self.relu(output_)
        output_ = self.downsample_4(output_)

        output_ = self.flatten(output_)

        output_ = self.fc_block(output_)

        

        return output_ # this is logit not probability
    
    def predict(self, x):
        output_ = self.forward(x)
        output_ = self.last_layer_activation(output_)
        return output_
    
    def initialize_parameters(self):
        for idx, m in enumerate(self.modules()):
            # print(idx, '->', m)
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                # print("ConvLayer or LinearLayer")
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                # print("BatchNormLayer")
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

class CnnWithResidualConnectionPTB(CnnWithResidualConnection):
    def __init__(self, config={"num_classes": 1}, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.num_classes = 1 # Binary classification for PTB dataset
        self.last_layer_activation = nn.Sigmoid()
    
    def forward(self, x):
        out_ =  super().forward(x)

        # converting to prob in `forward` function beacuse logits cannot
        # be passed to 
        out_ = self.last_layer_activation(out_)
        out_ = out_.squeeze() # for BCELoss size needs to be changed (array)
        return out_
    
    def predict(self, x):
        return self.forward(x)


class CnnWithResidualConnectionTransferMitbihToPtb(CnnWithResidualConnectionPTB):
    def __init__(self, config={ "num_classes": 1 }, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=False):
        new_state_dict = {}
        # filtering state dict to remove last layers:
        for name, weight in state_dict.items():
            if not name.startswith("fc_block"):
                new_state_dict[name] = weight
                
        super().load_state_dict(new_state_dict, strict=strict)
    
    def load_state_dict_for_eval(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict=strict)

class CnnWithResidualConnectionTransferMitbihToPtbFrozen(CnnWithResidualConnectionPTB):
    def __init__(self, config={ "num_classes": 1 }, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=False):
        new_state_dict = {}
        # filtering state dict to remove last layers:
        for name, weight in state_dict.items():
            if not name.startswith("fc_block"):
                new_state_dict[name] = weight
        
        # all weights are frozen except for the classification fc_block
        for name, param in self.named_parameters():
            if not name.startswith("fc_block") :
                param.requires_grad = False 
                
        super().load_state_dict(new_state_dict, strict=strict)
    
    def load_state_dict_for_eval(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict=strict)

        

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
                dropout_layer = nn.Dropout(p=0.2)
                current_block.append(relu_layer)
                current_block.append(dropout_layer)
        
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

        self.process_config(config=config)

        self._build_model()
    
    def process_config(self, config):
        self.default_config = {
            "num_input_channels": 1,

            
            "first_transformation": {
                # this is the number of channels of the output of the 
                # first conv and batchnrom operation. This output is 
                # further fed to the residual blocks,
                "num_output_channels": 16,
                "kernel_size": 3,
                "padding": 2,
                "stride": 1,
                # number of conv - bn - relu layers in this first block
                "num_conv_layers": 2
            },


            # this config is to control the general attributes of 
            # the residual block
            "residual_block_config": {
                # the stride to use when downsampling a map
                "downsampling_strides": [2, 2, 2, 2, 2],
                "kernel_sizes": [3, 3, 3, 3, 3],

                # this is the number of conv blocks used in each of the 
                # residual blocks. (It must be an even number)
                "num_conv_layers_in_blocks": [2, 2, 2, 2, 4],

                # it is a list of number of output channels for each of the
                # residual block (in order)
                "num_filters": [16, 32, 64, 128, 256],


            },

            "fully_connected_layer_input_size": 256,

            "num_classes": 5
        }

        # if config is provided then use that otherwise default one
        self.config = self.default_config
        if config is not None:
            self.config = config

    def forward(self, x) :
        out_ = self.intial_transformation(x)

        # apply residual blocks
        for block in self.residual_blocks:
            out_ = block(out_)

        # apply avegrage pooling
        out_ = self.pooling(out_)

        # flatten and feed to Fully connected layer
        out_ = self.flatten(out_)
        out_ = self.fc(out_)
        out_ = self.relu_op(out_)

        # do softmax
        out_ = self.softmax(out_)
        
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

            #we want shortcut connection after every second block
            shortcut_connection_flags = [1 if (idx+1)%2==0 else 0 for idx in 
                                            range(num_conv_blocks)]
            


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
        initial_transformation_layers = []
        current_input_channel = num_input_channels
        for idx in range(ft_config["num_conv_layers"]):
            initial_transformation_layers.append(
                self.get_conv_bn_relu_block(
                    in_channels=current_input_channel,
                    out_channels=ft_output_channels,
                    kernel_size=ft_kernel_size,
                    stride=ft_stride,
                    padding=ft_padding
                )
            )
            current_input_channel = ft_output_channels

        self.intial_transformation = nn.Sequential(
                                            *initial_transformation_layers)
        self.residual_blocks = self.get_residual_blocks(self.config,
                                        num_input_channels=ft_output_channels)

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(self.config["fully_connected_layer_input_size"],
                                self.config["num_classes"])
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.relu_op = nn.ReLU()



class CnnWithResidualBlocksPTB(CnnWithResidualBlocks):
    def process_config(self, config):
        super().process_config(config)
        self.config["num_classes"] = 2 # change the number of classes for PTB
        # keeo rest of the architecture same as CnnWithResidualBlocks
        


