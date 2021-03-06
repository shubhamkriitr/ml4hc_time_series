import torch

from model_cnn import (VanillaCnnPTB, VanillaCnnMITBIH)
from model_cnn_ae import (
 CnnPretrainEncoderWithTrainableClassifierHead, CnnEncoderDecoder,
 CnnEncoder, CnnPretrainEncoderWithTrainableClassifierHeadPTB,
 CnnPretrainEncoderWithTrainableClassifierHeadPartiallyFrozen
 )

from model_cnn_res import (CnnWithResidualConnection,
 CnnWithResidualConnectionPTB, CnnWithResidualConnectionTransferMitbihToPtb,
 CnnWithResidualConnectionTransferMitbihToPtbFrozen)

from model_rnn_b import (RnnModelPTB, RnnModelMITBIH,
 RnnModelMITBIHLongerSeq, VanillaRNNPTB, VanillaRNNMITBIH)
from model_cnn_2d import (CnnModel2DMITBIH, CnnModel2DPTB)
from model_transformer import (TransformerModelMITBIH, TransformerModelPTB)
from model_lstm import (BidirLstmModelMITBIH, BidirLstmModelPTB)

MODEL_CNN_ENCODER_DECODER = "CnnEncoderDecoder"
MODEL_CNN_ENCODER = "CnnEncoder"
MODEL_UNET_PRETRAINED_ENCODER_NN_CLASSIFIER = \
    "CnnPretrainEncoderWithTrainableClassifierHead"


MODEL_NAME_TO_CLASS_MAP = {
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
    "CnnModel2DPTB": CnnModel2DPTB,
    "TransformerModelMITBIH": TransformerModelMITBIH,
    "TransformerModelPTB": TransformerModelPTB,
    "CnnPretrainEncoderWithTrainableClassifierHeadPartiallyFrozen":\
         CnnPretrainEncoderWithTrainableClassifierHeadPartiallyFrozen,
    "RnnModelMITBIHLongerSeq": RnnModelMITBIHLongerSeq,
    "BidirLstmModelMITBIH": BidirLstmModelMITBIH,
    "BidirLstmModelPTB": BidirLstmModelPTB,
    "VanillaRNNPTB": VanillaRNNPTB,
    "VanillaRNNMITBIH": VanillaRNNMITBIH,
    "VanillaCnnPTB": VanillaCnnPTB,
    "VanillaCnnMITBIH": VanillaCnnMITBIH,
    "CnnWithResidualConnectionTransferMitbihToPtb":\
        CnnWithResidualConnectionTransferMitbihToPtb,
    "CnnWithResidualConnectionTransferMitbihToPtbFrozen":\
        CnnWithResidualConnectionTransferMitbihToPtbFrozen


}


MODEL_NAME_TO_WEIGHTS_PATH = {
    MODEL_UNET_PRETRAINED_ENCODER_NN_CLASSIFIER:\
         None,
    MODEL_CNN_ENCODER_DECODER: None,
    MODEL_CNN_ENCODER: None,
    "CnnWithResidualConnection": "saved_models/2022-03-28_000731__CnnWithResidualConnection/best_model.ckpt",
    "CnnPretrainEncoderWithTrainableClassifierHeadPTB": \
        None,
    "CnnWithResidualConnectionPTB": \
        "saved_models/2022-03-28_183658__exp_2_a_CnnWithResidualConnectionPTB/best_model.ckpt",
    "RnnModelPTB": "saved_models/2022-03-28_084444__RnnModelPTB/best_model.ckpt",
    "RnnModelMITBIH": "saved_models/2022-03-28_174957__RnnModelMITBIH/best_model.ckpt",
    "CnnModel2DMITBIH": "saved_models/2022-03-28_215502__exp_6_a_CnnModel2DMITBIH/best_model.ckpt",
    "CnnModel2DPTB": "saved_models/2022-03-28_224128__CnnModel2DPTB/best_model.ckpt",
    "TransformerModelMITBIH": None,
    "TransformerModelPTB": None,
    "CnnPretrainEncoderWithTrainableClassifierHeadPartiallyFrozen": None,
    "RnnModelMITBIHLongerSeq": None,
    "BidirLstmModelMITBIH": None,
    "BidirLstmModelPTB": None,
    "VanillaRNNMITBIH": "saved_models/2022-03-28_235015__exp_10_b_VanillaRNNMITBIH/best_model.ckpt",
    "VanillaRNNPTB": "saved_models/2022-03-28_234942__exp_10_a_VanillaRNNPTB/best_model.ckpt",
    "VanillaCnnPTB": "saved_models/2022-03-29_014835__exp_0_b_VanillaCnnPTB/best_model.ckpt",
    "VanillaCnnMITBIH": "saved_models/2022-03-29_012323__exp_0_a_VanillaCnnMITBIH/best_model.ckpt",
    "CnnWithResidualConnectionTransferMitbihToPtb": "saved_models/2022-03-29_202045__exp_11_a_CnnWithResidualConnectionTransferMitbihToPtb/best_model.ckpt",
    "CnnWithResidualConnectionTransferMitbihToPtbFrozen": "saved_models/2022-03-29_201122__exp_11_c_CnnWithResidualConnectionTransferMitbihToPtbFrozen/best_model.ckpt"

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
        if hasattr(model, "load_state_dict_for_eval"):
            model.load_state_dict_for_eval(state_dict=state_dict, strict=True)
        else:
            model.load_state_dict(state_dict=state_dict, strict=True)
        # make sure to call model.eval() or model.train() based on the usage
        return model

    def get_lazy_loader(self, model_name):
        return lambda : self.get(model_name)
    
    def load_from_location(self, model_name, model_weights_path):
        model_class =  super().get(model_name)

        model: torch.nn.Module = model_class() # Assumes model does not need init params

        state_dict = torch.load(model_weights_path)
        if hasattr(model, "load_state_dict_for_eval"):
            model.load_state_dict_for_eval(state_dict=state_dict, strict=True)
        else:
            model.load_state_dict(state_dict=state_dict, strict=True)
        # make sure to call model.eval() or model.train() based on the usage
        return model
        

if __name__ == "__main__":
    model_factory = TrainedModelFactory()
    model = model_factory.get("CnnWithResidualConnectionPTB")
    print(model)