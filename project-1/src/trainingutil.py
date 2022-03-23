import os
import logging
from argparse import ArgumentParser
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataLoaderUtil
from model_factory import ModelFactory
from util import get_timestamp_str

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
            self.invoke_epoch_callbacks(self.model, global_batch_number,
                                current_epoch, current_epoch_batch_number)
            
    def training_step(self, batch_data,  global_batch_number, current_epoch,
                    current_epoch_batch_number):
        
        # make one training step
        
        raise NotImplementedError()

    def invoke_epoch_callbacks(self, model, global_batch_number,
                                current_epoch, current_epoch_batch_number):
        self.invoke_callbacks(self.epoch_callbacks, 
                    [self.model, None, global_batch_number,
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
        train_shuffle=True, val_split=0.0)

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

        if self.config["load_from_checkpoint"]:
            checkpoint_path = self.config["checkpoint_path"]
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
            logger.info(str(self.model))
            logger.info(f"Model Loaded")
        
        self.model = model
        return model
    
    def prepare_optimizer(self):
        lr = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]
        self.optimizer = Adam(
            lr=lr, weight_decay=weight_decay,
            params=self.model.parameters()
            )

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
        if self.config["cost_function_class_name"] == "MSELoss":
            self.cost_function = nn.MSELoss()
        else:
            raise NotImplementedError()

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
            f"[{current_epoch}/{current_epoch_batch_number}]"
            f" Loss: {kwargs['loss']}")
        if global_batch_number % self.config["tensorboard_log_frequency"] == 0:
            self.summary_writer.add_scalar("train/loss", kwargs['loss'])
    
    def epoch_callback(self, model: nn.Module, batch_data, global_batch_number,
                    current_epoch, current_epoch_batch_number, **kwargs):
        file_path = get_timestamp_str()\
            + f"epoch_{current_epoch}_gbatch_{global_batch_number}.ckpt"
        # torch.save(model.state_dict(), file_path)
        
        model.eval()
        n_mc_samples = 1
        with torch.no_grad():
            
            x = self.test_loader.dataset.x
            y_true = self.test_loader.dataset.y

            y_pred_prob = model(x)/n_mc_samples
            for nmc in range(n_mc_samples -1):
                y_pred_prob = y_pred_prob + model(x)/n_mc_samples
            y_pred = y_pred_prob
            
            loss = self.cost_function(input=y_pred, target=y_true)

            print(f"Test loss: {loss}")
            self.summary_writer.add_scalar("test/loss", loss)


            

        



if __name__ == "__main__":
    DEFAULT_CONFIG_LOCATION = "experiment_configs/sample.yaml"
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str,
                            default=DEFAULT_CONFIG_LOCATION)
    args = argparser.parse_args()
    
    config_data = None
    with open(args.config, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    
    if config_data["pipeline_class"] == "ExperimentPipeline":
        pipeline = ExperimentPipeline(config=config_data)
        pipeline.prepare_experiment()
        pipeline.run_experiment()


