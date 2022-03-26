from torch import nn
import torch
import torch.functional as F


class RnnModelPTB (nn.Module):
    def __init__(self, config={"num_classes": 1}) -> None:
        super().__init__()
        # TODO: layer definition
        self.num_classes = config["num_classes"]
        self.linear = nn.Linear(in_features=187, out_features=self.num_classes)
        self.flatten = nn.Flatten()
        self.classification_activation_layer = nn.Sigmoid()
    
    def forward(self, x):
        # TODO: layer application
        out_ = self.flatten(x)
        out_ = self.linear(out_)

        # shape should be (batch, num_classes)
        out_ = self.classification_activation_layer(out_)
        # Shape should be (batch,)
        return self.reshape_output(out_)
    
    def reshape_output(self, out_):
        return out_.squeeze()

class RnnModelMITBIH(RnnModelPTB):
    def __init__(self, config={ "num_classes": 5 }) -> None:
        super().__init__(config)
        self.classification_activation_layer = nn.Softmax(dim=1)
    
    def reshape_output(self, out_):
        # shape does not need to changed for this
        return out_
