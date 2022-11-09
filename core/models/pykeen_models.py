from .base_model import *

class Pykeen_model(BaseKGE):
    
    def __init__(self,args,model) -> None:
        super.__init__(args)
        self.model=model
    

    def select_model(self):
        '''
        
        '''
        print("select pykeen model: ", self.model)