from model_cnn_ae import (UnetEncoder, UnetEncoderDecoder,
 CnnPretrainEncoderWithTrainableClassifierHead, CnnEncoderDecoder,
 CnnEncoder, CnnPretrainEncoderWithTrainableClassifierHeadPTB, 
 )

from model_cnn_res import (CnnWithResidualConnection,
 CnnWithResidualConnectionPTB)

from model_rnn_b import RnnModelPTB, RnnModelMITBIH

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
    "RnnModelMITBIH": RnnModelMITBIH

}

class ModelFactory(object):
    def get(self, model_name):
        return MODEL_NAME_TO_CLASS_MAP[model_name]
