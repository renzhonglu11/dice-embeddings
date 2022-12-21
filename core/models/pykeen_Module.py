import torch.utils.data
from pykeen import predict
import torch
import numpy as np
from typing import Tuple


class Pykeen_Module:
    def __init__(self, model_name) -> None:
        self.name = model_name
        

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

        if len(relation_embedd) == 1:
            relation_embedd = relation_embedd[0]

        if hasattr(self.model,"base") and hasattr(self.model.base,"entity_representations") and len(self.model.base.entity_representations)!=0:
            for embedd_item in self.model.base.entity_representations:
                entity_embedd.append(embedd_item().data.detach())


        if (
            hasattr(self.model, "entity_representations")
            and len(self.model.entity_representations) != 0
        ):
            for embedd_item in self.model.entity_representations:
                entity_embedd.append(embedd_item().data.detach())


     
        if len(entity_embedd) == 1:
            entity_embedd = entity_embedd[0]

        return (
            entity_embedd,
            relation_embedd,
        )


    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # the tensors here is inference tensors and can't be modified in-place outside InferenceMode.
        # https://twitter.com/PyTorch/status/1437838242418671620?s=20&t=8pEheJu4kRaLyJHBBLUvZA (solution)
        return predict.predict_triples(model=self.model, triples=x,).scores.clone()

