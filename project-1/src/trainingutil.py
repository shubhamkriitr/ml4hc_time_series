import os
import logging
from argparse import ArgumentParser
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import DataLoaderUtil, ClassWeights
from model_factory import ModelFactory
from util import get_timestamp_str
from metric_auroc_auprc import plot_auroc

logger = logging.getLogger(name=__name__)


class BaseTrainer(object):

    def __init__(self, model: nn.Module, dataloader, cost_function,
                 optimizer: Optimizer,
                batch_callbacks=[], epoch_callbacks=[], config={}) -> None:
        self.model = model
        self.cost_function = cost_function
        self.dataloader = dataloader
        self.batch_callbacks = batch_callbacks
        self.epoch_callbacks = epoch_callbacks

        # read from config: TODO
        self.num_epochs = 100
        self.config = config
        self.num_epochs = self.config["num_epochs"]
        self.optimizer = optimizer


    def train(self):
        global_batch_number = 0
        current_epoch_batch_number = 0
        for current_epoch in range(1, self.num_epochs + 1):
            current_epoch_batch_number = 0
            for batch_data in self.dataloader:
                global_batch_number += 1
                current_epoch_batch_number += 1

                # perform one training step
                self.training_step(batch_data, global_batch_number,
                                    current_epoch, current_epoch_batch_number)
            self.invoke_epoch_callbacks(self.model, batch_data, global_batch_number,
                                current_epoch, current_epoch_batch_number)
            
    def training_step(self, batch_data,  global_batch_number, current_epoch,
                    current_epoch_batch_number):
        
        # make one training step
        
        raise NotImplementedError()

    def invoke_epoch_callbacks(self, model, batch_data, global_batch_number,
                                current_epoch, current_epoch_batch_number):
        self.invoke_callbacks(self.epoch_callbacks, 
                    [self.model, batch_data, global_batch_number,
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


class BaseExperimentPipeline(object):
    """Class to link experiment stages like
    training, logging, evaluation, summarization etc.
    """

    def __init__(self, config) -> None:
        self.config = None
        self.initialize_config(config)
    
    def initialize_config(self, config):
        config = self.load_config(config)

        # TODO: add/ override some params here
        self.config = config


    def prepare_experiment(self):
        raise NotImplementedError()

    def run_experiment(self):
        raise NotImplementedError()

    def load_config(self, config):
        if isinstance(config, dict):
            return config
        if isinstance(config, str):
            config_data = {}
            with open(config, "r", encoding="utf-8") as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            return config_data

# dictionary to refer to class by name
# (to be used in config)
TRAINER_NAME_TO_CLASS_MAP = {
    "CnnTrainer": CnnTrainer
}

# Factory class to get trainer class by name
class TrainerFactory(object):
    def get(self, trainer_name):
        return TRAINER_NAME_TO_CLASS_MAP[trainer_name]



class ExperimentPipeline(BaseExperimentPipeline):
    def __init__(self, config) -> None:
        super().__init__(config)

    def prepare_experiment(self):
        self.prepare_model()
        self.prepare_optimizer() # call this after model has been initialized
        self.prepare_scheduler()
        self.prepare_cost_function()
        self.prepare_summary_writer()
        self.prepare_dataloaders()
        self.prepare_batch_callbacks()
        self.prepare_epoch_callbacks()

        self.trainer = self.prepare_trainer()


    def prepare_dataloaders(self):
        dataloader_class_name = self.config["dataloader_class_name"]
        train_batch_size = self.config["batch_size"]

        train_loader, val_loader, test_loader \
        = DataLoaderUtil().get_data_loaders(
        dataloader_class_name,
        train_batch_size=train_batch_size, 
        val_batch_size=1,
        test_batch_size=self.config["test_batch_size"], 
        train_shuffle=True, val_split=self.config["val_split"])

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def prepare_trainer(self):
        trainer_class = TrainerFactory().get(
            self.config["trainer_class_name"])
        
        trainer = trainer_class(model=self.model,
                    dataloader=self.train_loader,
                    cost_function=self.cost_function,
                    optimizer=self.optimizer,
                    batch_callbacks=self.batch_callbacks,
                    epoch_callbacks=self.epoch_callbacks,
                    config={
                        "num_epochs": self.config["num_epochs"]
                        }
                    )

        self.trainer = trainer
        return trainer
    

    def prepare_model(self):
        # TODO: use model config too (or make it work by creating new class)
        model = ModelFactory().get(self.config["model_class_name"])()
        self.model = model

        if self.config["load_from_checkpoint"]:
            checkpoint_path = self.config["checkpoint_path"]
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
            logger.info(str(self.model))
            logger.info(f"Model Loaded")
        
        return self.model
    
    def prepare_optimizer(self):
        trainable_params, trainable_param_names, frozen_params, \
                frozen_param_names = self.filter_trainer_parameters()
        print(f"Frozen Parameters: {frozen_param_names}")
        print(f"Trainable Parameters: {trainable_param_names} ")
        lr = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]
        optimizer_class = AdamW # Adam
        self.optimizer = optimizer_class(
            lr=lr, weight_decay=weight_decay,
            params=trainable_params
            )
    
    def prepare_scheduler(self):
        if "scheduler" not in self.config:
            return
        scheduler_name = self.config["scheduler"]
        if scheduler_name is None:
            return
        if scheduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        else:
            raise NotImplementedError()
        
    
    def filter_trainer_parameters(self):
        trainable_params = []
        trainable_param_names = []
        frozen_params = []
        frozen_param_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                trainable_param_names.append(name)
            else:
                frozen_params.append(param)
                frozen_param_names.append(name)
        
        return trainable_params, trainable_param_names, frozen_params, \
                frozen_param_names

    def prepare_summary_writer(self):
        experiment_tag = self.config["experiment_metadata"]["tag"]
        timestamp = get_timestamp_str()
        self.current_experiment_directory = os.path.join(
            self.config["logdir"],timestamp+"_"+experiment_tag)

        os.makedirs(self.current_experiment_directory, exist_ok=True)
        self.current_experiment_log_directory = os.path.join(
            self.current_experiment_directory, "logs"
        )
        os.makedirs(self.current_experiment_log_directory, exist_ok=True)
        
        self.summary_writer = SummaryWriter(
            log_dir=self.current_experiment_log_directory)

    def prepare_cost_function(self):
        class_weights = self.prepare_class_weights_for_cost_function()
        if class_weights is not None:
            print("Using class weights: ", class_weights)
        if self.config["cost_function_class_name"] == "MSELoss":
            print("Using: MSELoss")
            self.cost_function = nn.MSELoss()
        elif self.config["cost_function_class_name"] == "CrossEntropyLoss":
            print("Using: CrossEntropyLoss")
            self.cost_function = nn.CrossEntropyLoss(weight=class_weights)
        elif self.config["cost_function_class_name"] == "BCELoss":
            print("Using: BCELoss")
            self.cost_function = nn.BCELoss(weight=class_weights    )
        else:
            raise NotImplementedError()
    
    def prepare_class_weights_for_cost_function(self):
        if "do_class_weighting" in self.config and \
                self.config["do_class_weighting"]:
            # Based on the data loader being used and the weighting scheme,
            #  fetch the class weights
            return ClassWeights().get(
                self.config["dataloader_class_name"],
                self.config["class_weighting_scheme"])
        
        # None is the default value for most of the cost classes

        return None
        

    def prepare_batch_callbacks(self):
        self.batch_callbacks = [self.batch_callback]

    def prepare_epoch_callbacks(self):
        self.epoch_callbacks = [self.epoch_callback]

    def run_experiment(self):
        self.trainer.train()
    
    def batch_callback(self, model, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        
        if global_batch_number % self.config["batch_log_frequency"] == 0:
            print(
            f"[({global_batch_number}){current_epoch}-{current_epoch_batch_number}]"
            f" Loss: {kwargs['loss']}")
        if global_batch_number % self.config["tensorboard_log_frequency"] == 0:
            self.summary_writer.add_scalar("train/loss", kwargs['loss'], global_batch_number)
    
    def epoch_callback(self, model: nn.Module, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        if current_epoch == 1: # the epoch just finished
            # save the config
            self.save_config()
    
        model.eval()

    def save_config(self):
        try:
            file_path = os.path.join(self.current_experiment_directory,
                                    "config.yaml")
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f)
        except Exception as exc:
            print(exc) # TODO: replace all prints with logger         


class ExperimentPipelineForAutoEncoder(ExperimentPipeline):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.best_loss = 100000000000

    def epoch_callback(self, model: nn.Module, batch_data, global_batch_number,
     current_epoch, current_epoch_batch_number, **kwargs):
        super().epoch_callback(model, batch_data, global_batch_number, 
        current_epoch, current_epoch_batch_number, **kwargs)
        if current_epoch == 1:
            with torch.no_grad():
                self.summary_writer.add_graph(self.model, batch_data[0])
        

        # N.B. it is validation loss but it uses test dataloader
        val_loss = self.compute_and_log_evaluation_metrics(
            model, current_epoch, "val"
        )
        metric_name = "Validation loss"
        metric_value = val_loss

        print(f"{metric_name}: {metric_value}")

        if metric_value < self.best_loss:
            print(f"Saving model: {metric_name} changed from {self.best_loss}"
                  f" to {metric_value}")
            self.best_loss = metric_value
            file_path = os.path.join(self.current_experiment_directory,
            "best_model.ckpt")
            torch.save(model.state_dict(), file_path)
    
    def compute_and_log_evaluation_metrics(self, model, current_epoch,
        eval_type):
        if eval_type == "val" or eval_type == "test":
            x = self.test_loader.dataset.x
            y_true = self.test_loader.dataset.y
        
        y_pred = model(x)
        loss = self.cost_function(input=y_pred, target=y_true)

        self.summary_writer.add_scalar(f"{eval_type}/loss", loss, current_epoch)

        return loss



class ExperimentPipelineForClassifier(ExperimentPipeline):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.best_metric = None

    def epoch_callback(self, model: nn.Module, batch_data, global_batch_number,
     current_epoch, current_epoch_batch_number, **kwargs):
        if current_epoch == 1: # the epoch just finished
            # save the config
            self.save_config()
            with torch.no_grad():
                self.summary_writer.add_graph(self.model, batch_data[0])
    
        model.eval()
        # 

        with torch.no_grad():
            val_f1, _, _, _ = self.compute_and_log_evaluation_metrics(
                model, current_epoch, "val")
            test_f1, _, _, y_test_pred_prob = self.compute_and_log_evaluation_metrics(
                model, current_epoch, "test")
        
        metric_to_use_for_model_selection = val_f1 # TODO: can be pulled in config
        metric_name = "Validation F1-Score"
        if self.best_metric is None or \
             (self.best_metric < metric_to_use_for_model_selection):
            print(f"Saving model: {metric_name} changed from {self.best_metric}"
                  f" to {metric_to_use_for_model_selection}")
            self.best_metric = metric_to_use_for_model_selection
            file_path = os.path.join(self.current_experiment_directory,
            "best_model.ckpt")
            torch.save(model.state_dict(), file_path)

            self.generate_roc_curves(self.test_loader.dataset.y,
                 y_test_pred_prob)
        if hasattr(self, "scheduler"):
            self.scheduler.step(metric_to_use_for_model_selection)
            next_lr = [group['lr'] for group in self.optimizer.param_groups][0]
            self.summary_writer.add_scalar("lr", next_lr,
             current_epoch)
        return self.best_metric

    def compute_and_log_evaluation_metrics(self, model, current_epoch,
        eval_type):
        if eval_type == "test":
            x = self.test_loader.dataset.x
            y_true = self.test_loader.dataset.y
        if eval_type == "val":
            x = self.val_loader.dataset.x
            y_true = self.val_loader.dataset.y
        
        model_output = None
        try:
            model_output = model(x)
        except Exception as exc:
            print(exc)

        if hasattr(model, "predict"):
            y_pred_prob = model.predict(x)
            if model_output is None:
                model_output = y_pred_prob
        else:
            if model_output is None:
                model_output = model(x)
            y_pred_prob = model_output

        if "task_type" in self.config and \
                self.config["task_type"] == "binary_classification":
            if torch.max(y_pred_prob) > 1.0 or torch.min(y_pred_prob) < 0.:
                print("warning!: expected probability but received logits!")
            y_pred = (y_pred_prob>0.5).type(torch.int8)
        else:
            y_pred = torch.argmax(y_pred_prob, axis=1)
        f1 = f1_score(y_true, y_pred, average="macro")

        print("%s f1 score : %s "% (eval_type, f1))

        acc = accuracy_score(y_true, y_pred)
            
        loss = self.cost_function(input=model_output, target=y_true)
        print(f"{eval_type} acc: {acc}")
        print(f"{eval_type} loss: {loss}")
        self.summary_writer.add_scalar(f"{eval_type}/loss", loss, current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/F1", f1, current_epoch)
        self.summary_writer.add_scalar(f"{eval_type}/Accuracy", acc, current_epoch)
        self.summary_writer.flush()
        return f1, acc, loss, y_pred_prob
    
    def generate_roc_curves(self, y_true, y_pred_prob):
        if "generate_roc_curves" not in self.config:
            return
        if not self.config["generate_roc_curves"]:
            return
        save_path_prefix = os.path.join(self.current_experiment_log_directory,
             get_timestamp_str() + "_metrics_")
        print("Saving plots at "+save_path_prefix+"*")
        plot_auroc(y_true, y_pred_prob, save_path_prefix)
        

        
PIPELINE_NAME_TO_CLASS_MAP = {
    "ExperimentPipeline": ExperimentPipeline,
    "ExperimentPipelineForAutoEncoder": ExperimentPipelineForAutoEncoder,
    "ExperimentPipelineForClassifier": ExperimentPipelineForClassifier
}


if __name__ == "__main__":
    DEFAULT_CONFIG_LOCATION = "experiment_configs/experiment_0_a_vanilla_cnn_mitbih.yaml"
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str,
                            default=DEFAULT_CONFIG_LOCATION)
    args = argparser.parse_args()
    
    config_data = None
    with open(args.config, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    

    pipeline_class = PIPELINE_NAME_TO_CLASS_MAP[ config_data["pipeline_class"]]
    pipeline = pipeline_class(config=config_data)
    pipeline.prepare_experiment()
    pipeline.run_experiment()


