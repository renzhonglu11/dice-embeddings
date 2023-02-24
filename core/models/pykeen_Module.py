import torch.utils.data
from pykeen import predict
import torch
import numpy as np
from typing import Dict, Tuple
import pandas as pd
from .base_model import *


class Pykeen_Module:
    def __init__(self, model_name,optimizer) -> None:
        self.name = model_name
        self.selected_optimizer = optimizer

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        relation_embedd = []
        entity_embedd = []


        if hasattr(self.model,"base") and hasattr(self.model.base,"relation_representations") and len(self.model.base.relation_representations)!=0:
            for embedd_item in self.model.base.relation_representations:
                relation_embedd.append(embedd_item().data.detach())

        if (
            hasattr(self.model, "relation_representations")
            and len(self.model.relation_representations) != 0
        ):
            for embedd_item in self.model.relation_representations:
                relation_embedd.append(embedd_item().data.detach())

        # if len(relation_embedd) == 1:
        #     relation_embedd = relation_embedd[0]

        if hasattr(self.model,"base") and hasattr(self.model.base,"entity_representations") and len(self.model.base.entity_representations)!=0:
            for embedd_item in self.model.base.entity_representations:
                entity_embedd.append(embedd_item().data.detach())


        if (
            hasattr(self.model, "entity_representations")
            and len(self.model.entity_representations) != 0
        ):
            for embedd_item in self.model.entity_representations:
                entity_embedd.append(embedd_item().data.detach())


     
        # if len(entity_embedd) == 1:
        #     entity_embedd = entity_embedd[0]

        return (
            entity_embedd,
            relation_embedd,
        )


    def forward_triples(self, x: torch.Tensor,h_prediction=False,t_prediction=False) -> torch.FloatTensor:
        # the tensors here is inference tensors and can't be modified in-place outside InferenceMode.
        # https://twitter.com/PyTorch/status/1437838242418671620?s=20&t=8pEheJu4kRaLyJHBBLUvZA (solution)
        # torch_max_mem will be used by default. If the tensors are not moved to cuda, a warning will occur
        # https://pykeen.readthedocs.io/en/latest/reference/predict.html#predict-triples-df (migration guide)
        
        if t_prediction:
            # torch.tensor(predictions_tails.df.score)

            predictions_tails = predict.predict_target(model=self.model,head = x[0,0].item(),relation = x[0,1].item(),targets=x[:,2])
          
            _df = predictions_tails.df.sort_values(by=['tail_id'])
            return torch.tensor(_df.score,dtype=torch.float64)
            # tensor = self.model.score_t(x[:0,:1],tails = x[:2])
            # return tensor
        if h_prediction:
            predictions_heads = predict.predict_target(model=self.model, relation = x[0,1].item(),tail=x[0,2].item(),targets=x[:,0])
            # for row in predictions_heads.df.iterrows():
            #     lst[int(row['head_id'])] = row['score']
            _df = predictions_heads.df.sort_values(by=['head_id'])
            # return lst
            return torch.tensor(_df.score,dtype=torch.float64)
        
        # return predict.predict_triples(model=self.model, triples=x.to("cuda"),).scores.clone()

    def mem_of_model(self) -> Dict:
        """ Size of model in MB and number of params"""
        # https://discuss.pytorch.org/t/finding-model-size/130275/2
        # (2) Store NumParam and EstimatedSizeMB
        num_params = sum(p.numel() for p in self.parameters())
        # Not quite sure about EstimatedSizeMB ?
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return {'EstimatedSizeMB': (num_params + buffer_size) / 1024 ** 2, 'NumParam': num_params}
