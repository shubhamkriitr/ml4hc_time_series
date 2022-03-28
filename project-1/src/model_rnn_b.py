from torch import nn
import torch
import torch.functional as F


class RnnModelMITBIH (nn.Module):
    def __init__(self, config={"num_classes": 5}) -> None:
        super().__init__()
        # TODO: layer definition
        self.num_classes = config["num_classes"]

        #network level hyper parameters 
        self.dropout = 0.0
        self.bidirectional = True

        # each of the vector passed to the RNN would have this many number
        # of elements
        self.input_feature_chunk_size = 21
        self.input_original_feature_size = 187
        self.last_layer_activation = nn.Softmax(dim=1)


        self._compute_and_initialize_sequence_length_and_padding()
        self._build_network()

    def _compute_and_initialize_sequence_length_and_padding(self):
        remainder \
            = self.input_original_feature_size % self.input_feature_chunk_size
        padding = 0
        sequence_length \
            = self.input_original_feature_size // self.input_feature_chunk_size
        if remainder > 0:
            sequence_length += 1
            padding = self.input_feature_chunk_size - remainder
        
        self.sequence_length = sequence_length
        self.padding = padding


    def _build_network(self):
        """Create the network layers.
        Also override this function in child classes to change network
        level configurations before the netwok is built.
        """
        self.rnn_block_0 = nn.RNN(
            input_size=self.input_feature_chunk_size,
            hidden_size=32,
            num_layers=3,
            nonlinearity='relu',
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        feature_factor = 2 if self.bidirectional else 1

        self.rnn_block_1 = nn.RNN(
            input_size=32*feature_factor,
            hidden_size=16,
            num_layers=4,
            nonlinearity='relu',
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        self.rnn_block_2 = nn.RNN(
            input_size=16*feature_factor,
            hidden_size=16,
            num_layers=4,
            nonlinearity='relu',
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        self.rnn_block_3 = nn.RNN(
            input_size=16*feature_factor,
            hidden_size=16,
            num_layers=4,
            nonlinearity='relu',
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        self.classification_head = self._create_classification_head()


    def _create_classification_head(self):
        """
        Create classification head which takes feature
        extracted by the RNN as input. 
        This function must be called inside `self._build_network`
        """
        feature_factor = 2 if self.bidirectional else 1
        in_features = self.sequence_length*16*feature_factor
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.num_classes),
            
        )


    def adjust_input(self, x):
        out_ = torch.transpose(x, 1, 2) # bring temporal dim 
        # to 2nd position (i.e. 1st index). e.g. B x 187 x 1

        if self.padding != 0:
            size_ = list(out_.shape)
            size_[1] = self.padding
            zero_pad = torch.zeros(size=size_, dtype=out_.dtype)
            out_ = torch.cat([out_, zero_pad], dim=1)

        if self.input_feature_chunk_size != 1:
            new_shape = (out_.shape[0],
             self.sequence_length,
             self.input_feature_chunk_size)

            out_ = out_.reshape(new_shape)
        
        return out_

        
    
    def forward(self, x):
        # TODO: layer application
        
        # format ouput corerctly to feed it to the network
        out_ = self.adjust_input(x)
        
        # Apply RNN layers
        out_, hidden_ = self.rnn_block_0(out_)
        out_, hidden_ = self.rnn_block_1(out_)
        out_, hidden_ = self.rnn_block_2(out_)
        out_, hidden_ = self.rnn_block_3(out_)

        out_ = self.classification_head(out_)




        # Shape should be (batch,) for binary classification case
        # and (batch, num_classes) otherwise
        # N.B. it is logits
        return self.reshape_output(out_)

    def predict(self, x):
        out_ = self.forward(x)
        out_ = self.last_layer_activation(out_)
        return out_
    
    def reshape_output(self, out_):
        # shape does not need to changed for this
        return out_
    
    

class RnnModelPTB(RnnModelMITBIH):
    def __init__(self, config={ "num_classes": 1 }) -> None:
        super().__init__(config)
    
    def _build_network(self):
        self.last_layer_activation = nn.Sigmoid()
        super()._build_network()
    
    
    def forward(self, x):
        logits = super().forward(x)
        return self.last_layer_activation(logits)

    def predict(self, x):
        return self.forward(x)
    
    def reshape_output(self, out_):
        return out_.squeeze()

if __name__ == "__main__":
    n_batch = 3
    n_feat = 187
    x = torch.arange(n_batch*n_feat).reshape(n_batch,1,n_feat).type(torch.float32)
    net = RnnModelMITBIH()
    y = net(x)
