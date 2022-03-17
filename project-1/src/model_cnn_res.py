from torch import nn
from torch import functional as F
import numpy as np
from data_loader import (MITBIHDataLoader, PTBDataLoader, DataLoaderUtil,
                            DATA_MITBIH, DATA_PTBDB, DataLoaderUtilMini)
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam

import copy
import logging
from sklearn.metrics import accuracy_score, f1_score
from util import get_timestamp_str

logger = logging.getLogger(name=__name__)

MODEL_CNN_RES = "CNN with Residual Blocks"




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
            dropout_layer = nn.Dropout(p=0.2)

            current_block.append(conv_layer)
            current_block.append(batchnorm_layer)
            current_block.append(dropout_layer)

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
                "downsampling_strides": [2, 2, 2],
                "kernel_sizes": [3, 3, 3],

                # this is the number of conv blocks used in each of the 
                # residual blocks. (It must be an even number)
                "num_conv_layers_in_blocks": [2, 2, 2],

                # it is a list of number of output channels for each of the
                # residual block (in order)
                "num_filters": [32, 64, 128],


            },

            "fully_connected_layer_input_size": 128,

            "num_classes": 5
        }

        # if config is provided then use that otherwise default one
        self.config = self.default_config
        if config is not None:
            self.config = config

        self._build_model()

    def forward(self, x) :
        out_ = self.intial_transformation(x)

        # apply residual blocks
        for block in self.residual_blocks:
            out_ = block(out_)

        # apply avegrage pooling
        out_ = self.average_pooling(out_)

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

        self.average_pooling = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(self.config["fully_connected_layer_input_size"],
                                self.config["num_classes"])
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.relu_op = nn.ReLU()



    

class BaseTrainer(object):

    def __init__(self, model: nn.Module, dataloader, cost_function,
                 optimizer: Optimizer,
                batch_callbacks=[], epoch_callbacks=[]) -> None:
        self.model = model
        self.cost_function = cost_function
        self.dataloader = dataloader
        self.batch_callbacks = batch_callbacks
        self.epoch_callbacks = epoch_callbacks

        # read from config: TODO
        self.max_epoch = 100
        # self.batch_size = 16
        self.lr = 1e-3
        self.optimizer = optimizer


    def train(self):
        global_batch_number = 0
        current_epoch_batch_number = 0
        for current_epoch in range(1, self.max_epoch + 1):
            current_epoch_batch_number = 0
            for batch_data in self.dataloader:
                global_batch_number += 1
                current_epoch_batch_number += 1

                # perform one training step
                self.training_step(batch_data, global_batch_number,
                                    current_epoch, current_epoch_batch_number)
            self.invoke_epoch_callbacks(self.model, global_batch_number,
                                current_epoch, current_epoch_batch_number)
            
    def training_step(self, batch_data,  global_batch_number, current_epoch,
                    current_epoch_batch_number):
        
        # make one training step
        
        raise NotImplementedError()

    def invoke_epoch_callbacks(self, model, global_batch_number,
                                current_epoch, current_epoch_batch_number):
        self.invoke_callbacks(self.epoch_callbacks, 
                    [self.model, None, global_batch_number,
                    current_epoch, current_epoch_batch_number], {})

    def invoke_callbacks(self, callbacks, args: list, kwargs: dict):
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                logger.error(exc)

class CnnTrainer(BaseTrainer):

    def training_step(self, batch_data, global_batch_number,
                        current_epoch, current_epoch_batch_number):
        # make sure training mode is on 
        self.model.train()

        # reset optimizer
        self.optimizer.zero_grad()

        # unpack batch data
        x, y_true = batch_data

        # compute model prediction
        y_pred = self.model(x)

        # compute loss
        loss = self.cost_function(input=y_pred, target=y_true)

        # backward pass
        loss.backward()

        # take optimizer step
        self.optimizer.step()

        self.invoke_callbacks(self.batch_callbacks, 
                    [self.model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number], {"loss": loss})


class CnnWithResidualBlocksPTB(CnnWithResidualBlocks):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config["num_classes"] = 2
        

def do_sample_training():
    cost_function = nn.CrossEntropyLoss()
    dataloader_util = DataLoaderUtil()

    dataset_name = DATA_PTBDB # or DATA_MITBIH
    model_class = CnnWithResidualBlocksPTB # or CnnWithResidualBlocks

    train_loader, val_loader, test_loader \
        = dataloader_util.get_data_loaders(dataset_name, train_batch_size=100, 
        val_batch_size=1, test_batch_size=100, train_shuffle=True,
        val_split=0.1)
    model = model_class(config=None) # using default model config
    opti = Adam(lr=1e-3, params=model.parameters(), weight_decay=1e-4)

    def batch_callback(model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        print(
            f"[{current_epoch}/{current_epoch_batch_number}]"
            f" Loss: {kwargs['loss']}")
    
    def save_model_callback(model: nn.Module, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        file_path = get_timestamp_str()\
            + f"epoch_{current_epoch}_gbatch_{global_batch_number}.ckpt"
        torch.save(model.state_dict(), file_path)
        
        model.eval()

        num_batches = 0
        acc_sum = 0

        n_mc_samples = 4
        with torch.no_grad():
            for i in [1]: #x, y_true in train_loader:
                x = test_loader.dataset.x
                y_true = test_loader.dataset.y

                y_pred_prob = model(x)/n_mc_samples
                for nmc in range(n_mc_samples -1):
                    y_pred_prob = y_pred_prob + model(x)/n_mc_samples
                y_pred = torch.argmax(y_pred_prob, axis=1)
                f1 = f1_score(y_true, y_pred, average="macro")

                print("Test f1 score : %s "% f1)

                acc = accuracy_score(y_true, y_pred)

                print("Test accuracy score : %s "% acc)
                num_batches += 1
                acc_sum += acc
        # print(f"Average Accuracy :  {acc_sum/num_batches}")
        

    trainer = CnnTrainer(model=model, dataloader=train_loader,
                cost_function=cost_function, optimizer=opti,
                 batch_callbacks=[batch_callback], 
                 epoch_callbacks=[save_model_callback])
    
    

    trainer.train()





if __name__ == "__main__":
    output_channels = [16, 16, 16, 16]
    kernel_sizes = [3 for i in range(len(output_channels))]
    paddings = [1 for i in range(len(output_channels))]
    strides = [2, 1, 1, 1]
    shortcut_connection_flags = [0, 1]*2

    res_block = ResidualBlock(1, output_channels, kernel_sizes, strides,
                     paddings, shortcut_connection_flags, True)

    print(res_block)

    do_sample_training()