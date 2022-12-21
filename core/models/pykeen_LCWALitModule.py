from pykeen.contrib.lightning import LCWALitModule
import torch
from .pykeen_Module import *

class MyLCWALitModule(LCWALitModule,Pykeen_Module):
    def __init__(self, *, model_name: str, **kwargs):
        Pykeen_Module.__init__(self,model_name)
        super().__init__(**kwargs)
        self.loss_history = []


    def training_epoch_end(self, training_step_outputs) -> None:
        batch_losses = [i["loss"].item() for i in training_step_outputs]
        avg = sum(batch_losses) / len(batch_losses)
        self.loss_history.append(avg)


  