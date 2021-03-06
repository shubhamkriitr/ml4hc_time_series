from model_factory import TrainedModelFactory
from metric_auroc_auprc import plot_auroc
from argparse import ArgumentParser
from data_loader import (MITBIH, PTBDB, DATA_MITBIH, DATA_PTBDB,
                         DataLoaderUtil)
from sklearn.metrics import accuracy_score, f1_score
import torch

dataloader_tag_to_name = {
    MITBIH: DATA_MITBIH,
    PTBDB: DATA_PTBDB
}

data_choices = list(dataloader_tag_to_name.keys())

model_choices = [
    "CnnWithResidualConnection", # This is for MITBIH
    "CnnWithResidualConnectionPTB",
    "RnnModelPTB", # this is bidirectional RNN
    "RnnModelMITBIH",
    "CnnModel2DMITBIH",
    "CnnModel2DPTB",
    "VanillaRNNPTB",
    "VanillaRNNMITBIH",
    "VanillaCnnMITBIH",
    "VanillaCnnPTB",
    "CnnWithResidualConnectionTransferMitbihToPtb",
    "CnnWithResidualConnectionTransferMitbihToPtbFrozen"
]


class ModelEvaluator:
    def __init__(self) -> None:
        pass

    def evaluate(self, model, dataset_tag, mode, count_params):
        if count_params:
            print(f"Total num. of trainable parameters in the model:"
                    f" {self.count_number_of_trainable_params(model)}")
            print(f"Total num. of  parameters in the model:"
                    f" {self.count_number_of_params(model)}")
        
        _, val_loader, test_loader\
             = DataLoaderUtil().get_data_loaders(
                 dataset_name=dataloader_tag_to_name[dataset_tag],
                 val_split=0.2
        )

        with torch.no_grad():

            self.evaluate_one(dataset_tag, model, test_loader.dataset.x, 
                test_loader.dataset.y, "test")

            self.evaluate_one(dataset_tag, model, val_loader.dataset.x,
                val_loader.dataset.y, "val")
        
        

    def evaluate_one(self, dataset_tag, model, x, y_true, tag_):
        model.eval()
        
        if hasattr(model, "predict"):
            y_pred_prob = model.predict(x)
        else:
            y_pred_prob = model(x)
        
        if dataset_tag == PTBDB:
            y_pred = (y_pred_prob>0.5).type(torch.int8)
        else:
            y_pred = torch.argmax(y_pred_prob, axis=1)

        f1 = f1_score(y_true, y_pred, average="macro")

        acc = accuracy_score(y_true, y_pred)

        self.print_scores(tag_, f1=f1, acc=acc)

        if dataset_tag == PTBDB:
            plot_auroc(y_true, y_pred_prob, figure_save_location_prefix=None,
                plot_tag=f"[{tag_}]")

    def print_scores(self, tag_, f1, acc):
        if tag_ == "test":
            tag_ = "Scores on test Data:"
        elif tag_ == "val":
            tag_ = "Scores on validation data:"

        
        print(f"{tag_}: Accuracy: {acc}, F1: {f1}")
    

    def count_number_of_params(self, model):
        return sum(p.numel() for p in model.parameters())
    
    def count_number_of_trainable_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=model_choices, required=True)
    parser.add_argument("--data", type=str, choices=data_choices, 
                        required=True)
    parser.add_argument("--mode", nargs="+", type=str, required=False,
                         default=["test", "val"])
    parser.add_argument("--count-params", action="store_true", default=False)
    parser.add_argument("--model-path", type=str, default="")

    args = parser.parse_args()

    model = None
    if args.model_path == "":
        lazy_model_loader = TrainedModelFactory().get_lazy_loader(args.model)
        model = lazy_model_loader()
        # The above two lines are equivalent to
        # model = TrainedModelFactory().get(args.model)
        # but model is not actually loaded until lazy_model_loader is called
    else:
        print(f"Loading weights from: {args.model_path}")
        model = TrainedModelFactory().load_from_location(
                        model_name=args.model,
                        model_weights_path=args.model_path)
    

    evaluator = ModelEvaluator()

    evaluator.evaluate(model, args.data, args.mode, args.count_params)