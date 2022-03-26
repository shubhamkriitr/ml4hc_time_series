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
    "CnnWithResidualConnection",
    "CnnWithResidualConnectionPTB"
]


class ModelEvaluator:
    def __init__(self) -> None:
        pass

    def evaluate(self, model, dataset_tag, mode):
        
        _, val_loader, test_loader\
             = DataLoaderUtil().get_data_loaders(
                 dataset_name=dataloader_tag_to_name[dataset_tag]
        )

        self.evaluate_one(dataset_tag, model, test_loader.dataset.x, 
            test_loader.dataset.y, "test")

        self.evaluate_one(dataset_tag, model, val_loader.dataset.x,
            val_loader.dataset.y, "val")
        
        

    def evaluate_one(self, dataset_tag, model, x, y_true, tag_):
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
                plot_tag=tag_)

    def print_scores(self, tag_, f1, acc):
        if tag_ == "test":
            tag_ = "Scores on test Data:"
        elif tag_ == "val":
            tag_ = "Scores on validation data:"

        
        print(f"{tag_}: Accuracy: {acc}, F1: {f1}")
        
        




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=data_choices, required=True)
    parser.add_argument("--data", type=str, choices=data_choices, 
                        required=True)
    parser.add_argument("--mode", nargs="+", type=str, required=False,
                         default=["test", "val"])

    args = parser.parse_args()

    model = TrainedModelFactory().get(args.model)

    evaluator = ModelEvaluator()

    evaluator.evaluate(model, args.data, args.mode)