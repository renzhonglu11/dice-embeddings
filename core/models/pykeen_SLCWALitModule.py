from pykeen.contrib.lightning import SLCWALitModule
from .pykeen_Module import *
from pykeen.triples.triples_factory import CoreTriplesFactory


class MySLCWALitModule(SLCWALitModule, Pykeen_Module):
    def __init__(self, *, model_name: str, args, **kwargs):
        Pykeen_Module.__init__(self, model_name)
        # import pdb; pdb.set_trace()
        super().__init__(**kwargs)
        self.loss_history = []
        self.args=args


    def training_epoch_end(self, training_step_outputs) -> None:
        batch_losses = [i["loss"].item() for i in training_step_outputs]
        avg = sum(batch_losses) / len(batch_losses)
        self.loss_history.append(avg)


