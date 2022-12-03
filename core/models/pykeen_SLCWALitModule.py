from pykeen.contrib.lightning import SLCWALitModule
from pykeen import predict
import torch
import numpy as np
from .pykeen_Module import *


class MySLCWALitModule(SLCWALitModule, Pykeen_Module):
    def __init__(self, *, model_name: str, **kwargs):
        Pykeen_Module.__init__(self, model_name)
        super().__init__(**kwargs)

