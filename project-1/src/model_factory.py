import torch
from model_cnn_ae import (UnetEncoder, UnetEncoderDecoder,
 CnnPretrainEncoderWithTrainableClassifierHead, CnnEncoderDecoder,
 CnnEncoder, CnnPretrainEncoderWithTrainableClassifierHeadPTB, 
 )

from model_cnn_res import (CnnWithResidualConnection,
 CnnWithResidualConnectionPTB)

from model_rnn_b import RnnModelPTB, RnnModelMITBIH
from model_cnn_2d import (CnnModel2DMITBIH)
from model_transformer import (TransformerModelMITBIH, TransformerModelPTB)


MODEL_UNET_ENCODER = "UnetEncoder"
MODEL_UNET_ENCODER_DECODER = "UnetEncoderDecoder"
MODEL_CNN_ENCODER_DECODER = "CnnEncoderDecoder"
MODEL_CNN_ENCODER = "CnnEncoder"
MODEL_UNET_PRETRAINED_ENCODER_NN_CLASSIFIER = \
    "CnnPretrainEncoderWithTrainableClassifierHead"


MODEL_NAME_TO_CLASS_MAP = {
    MODEL_UNET_ENCODER: UnetEncoder,
    MODEL_UNET_ENCODER_DECODER: UnetEncoderDecoder,
    MODEL_UNET_PRETRAINED_ENCODER_NN_CLASSIFIER:\
         CnnPretrainEncoderWithTrainableClassifierHead,
    MODEL_CNN_ENCODER_DECODER: CnnEncoderDecoder,
    MODEL_CNN_ENCODER: CnnEncoder,
    "CnnWithResidualConnection": CnnWithResidualConnection,
    "CnnPretrainEncoderWithTrainableClassifierHeadPTB": \
        CnnPretrainEncoderWithTrainableClassifierHeadPTB,
    "CnnWithResidualConnectionPTB": \
        CnnWithResidualConnectionPTB,
    "RnnModelPTB": RnnModelPTB,
    "RnnModelMITBIH": RnnModelMITBIH,
    "CnnModel2DMITBIH": CnnModel2DMITBIH,
    "TransformerModelMITBIH": TransformerModelMITBIH,
    "TransformerModelPTB": TransformerModelPTB

}


MODEL_NAME_TO_WEIGHTS_PATH = {
    MODEL_UNET_ENCODER: None,
    MODEL_UNET_ENCODER_DECODER: None,
    MODEL_UNET_PRETRAINED_ENCODER_NN_CLASSIFIER:\
         None,
    MODEL_CNN_ENCODER_DECODER: None,
    MODEL_CNN_ENCODER: None,
    "CnnWithResidualConnection": None,
    "CnnPretrainEncoderWithTrainableClassifierHeadPTB": \
        None,
    "CnnWithResidualConnectionPTB": \
        "saved_models/2022-03-26_203707__CnnWithResidualConnectionPTB/best_model.ckpt",
    "RnnModelPTB": None,
    "RnnModelMITBIH": None,
    "CnnModel2DMITBIH": None,
    "TransformerModelMITBIH": None,
    "TransformerModelPTB": TransformerModelPTB
}
class ModelFactory(object):
    def get(self, model_name):
        return MODEL_NAME_TO_CLASS_MAP[model_name]


class TrainedModelFactory(ModelFactory):
    def __init__(self, config = {}) -> None:
        super().__init__()
        # if config has `model_name_to_weights_path`
        self.config = config
        if "model_name_to_weights_path" not in self.config:
            self.config["model_name_to_weights_path"] \
                = MODEL_NAME_TO_WEIGHTS_PATH

        self.model_weights_path = self.config["model_name_to_weights_path"]
    
    def get(self, model_name):
        model_class =  super().get(model_name)
        model_weights_path = self.model_weights_path[model_name]

        model: torch.nn.Module = model_class() # Assumes model does not need init params

        state_dict = torch.load(model_weights_path)
        model.load_state_dict(state_dict=state_dict, strict=True)
        # make sure to call model.eval() or model.train() based on the usage
        return model
        

if __name__ == "__main__":
    model_factory = TrainedModelFactory()
    model = model_factory.get("CnnWithResidualConnectionPTB")
    print(model)