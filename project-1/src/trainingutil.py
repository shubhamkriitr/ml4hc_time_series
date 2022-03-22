import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import logging
from sklearn.metrics import accuracy_score, f1_score
from data_loader import DataLoaderUtil
from util import get_timestamp_str
from model_factory import (MODEL_UNET_ENCODER, MODEL_UNET_ENCODER_DECODER,
                            ModelFactory)
import yaml
from argparse import ArgumentParser

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
        if "num_epochs" in self.config:
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
                config_data = yaml.load(f)
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
        self.trainer = self.prepare_trainer()
        self.model = self.prepare_model()
        self.optimizer = self.prepare_optimizer() # call this after model has been initialized
        cost_function = self.prepare_cost_function()
        self.summary_writer = self.prepare_summary_writer()

        train_loader, val_loader, test_loader = self.prepare_dataloaders()

        self.trainer.train(
            model= self.model,
            dataloader=train_loader


        )

    def prepare_dataloaders(self):
        dataset_class_name = self.config["dataset_class_name"]
        train_batch_size = self.config["batch_size"]

        train_loader, val_loader, test_loader \
        = DataLoaderUtil().get_data_loaders(
        dataset_class_name,
        train_batch_size=train_batch_size, 
        val_batch_size=1,
        test_batch_size=self.config["test_batch_size"], 
        train_shuffle=True, val_split=0.0)

        return train_loader, val_loader, test_loader

    def prepare_trainer(self):
        trainer = TrainerFactory().get(
            self.config["trainer_class_name"])
        
        return trainer
    

    def prepare_model(self):
        model = ModelFactory().get(self.config["model_class_name"])

        if self.config["load_from_checkpoint"]:
            checkpoint_path = self.config["checkpoint_path"]
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
            logger.info(str(self.model))
            logger.info(f"Model Loaded")
        
        return model
    
    def prepare_optimizer(self):
        pass

    def prepare_summary_writer(self):
        pass

    def prepare_cost_function(self):
        pass

    def prepare_batch_callbacks(self):
        pass

    def prepare_epoch_callbacks(self):
        pass

    def run_experiment(self):
        return super().run_experiment()
            

        



if __name__ == "__main__":
    DEFAULT_CONFIG_LOCATION = "experiment_configs/sample.yaml"
    argparser = ArgumentParser()
    argparser.add_argument("--config", type="str",
                            default=DEFAULT_CONFIG_LOCATION)
