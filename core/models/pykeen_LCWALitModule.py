from pykeen.contrib.lightning import LCWALitModule
import torch
from .pykeen_Module import *

class MyLCWALitModule(LCWALitModule,Pykeen_Module):
    def __init__(self, *, model_name: str, **kwargs):
        Pykeen_Module.__init__(self,model_name)
        super().__init__(**kwargs)


  