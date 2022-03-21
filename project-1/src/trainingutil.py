import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import logging
from sklearn.metrics import accuracy_score, f1_score
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
        self.max_epoch = 100
        self.config = config
        if "max_epoch" in self.config:
            self.max_epoch = self.config["max_epoch"]
        self.optimizer = optimizer


    def train(self):
        global_batch_number = 0
        current_epoch_batch_number = 0
        for current_epoch in range(1, self.max_epoch + 1):
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