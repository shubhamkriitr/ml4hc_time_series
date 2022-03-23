from model_cnn_ae import UnetEncoder, UnetEncoderDecoder

MODEL_UNET_ENCODER = "UnetEncoder"
MODEL_UNET_ENCODER_DECODER = "UnetEncoderDecoder"


MODEL_NAME_TO_CLASS_MAP = {
    MODEL_UNET_ENCODER: UnetEncoder,
    MODEL_UNET_ENCODER_DECODER: UnetEncoderDecoder
}

class ModelFactory(object):
    def get(self, model_name):
        return MODEL_NAME_TO_CLASS_MAP[model_name]
