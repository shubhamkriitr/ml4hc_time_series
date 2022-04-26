## Config#1 : CNN With Residual Blocks

- Achieved scores after 10 epochs
```
Loss: 1.0352399349212646
Test f1 score : 0.32360733818222964 
Test accuracy score : 0.8946190389183263 
Average Accuracy :  0.8946190389183263
```

```
class CnnWithResidualBlocks(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.default_config = {
            "num_input_channels": 1,

            
            "first_transformation": {
                # this is the number of channels of the output of the 
                # first conv and batchnrom operation. This output is 
                # further fed to the residual blocks,
                "num_output_channels": 8,
                "kernel_size": 5,
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
                "num_filters": [16, 32, 64],


            },

            "fully_connected_layer_input_size": 64,

            "num_classes": 5
        }
```

# Config#2 

```
Test f1 score : 0.5737887392991092 
Test accuracy score : 0.9576557646628906
```

```
self.default_config = {
            "num_input_channels": 1,

            
            "first_transformation": {
                # this is the number of channels of the output of the 
                # first conv and batchnrom operation. This output is 
                # further fed to the residual blocks,
                "num_output_channels": 16,
                "kernel_size": 5,
                "padding": 2,
                "stride": 1,
                # number of conv - bn - relu layers in this first block
                "num_conv_layers": 3
            },


            # this config is to control the general attributes of 
            # the residual block
            "residual_block_config": {
                # the stride to use when downsampling a map
                "downsampling_strides": [2, 2, 2],
                "kernel_sizes": [3, 3, 3],

                # this is the number of conv blocks used in each of the 
                # residual blocks. (It must be an even number)
                "num_conv_layers_in_blocks": [4, 4, 4],

                # it is a list of number of output channels for each of the
                # residual block (in order)
                "num_filters": [32, 64, 128],


            },

            "fully_connected_layer_input_size": 128,

            "num_classes": 5
        }
```