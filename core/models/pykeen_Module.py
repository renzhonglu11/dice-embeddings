import torch.utils.data
from pykeen import predict
import torch
import numpy as np
from typing import Tuple
from pykeen.nn.representation import SingleCompGCNRepresentation


class Pykeen_Module:
    def __init__(self, model_name) -> None:
        self.name = model_name
        self.loss_history = []

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        entity_embedd = None
        relation_embedd = None

        # TODO: NodePiece uses selected anchor nodes and relation to embedd the target node. As a result not all nodes need to be embedded
        # if self.name.strip() == "NodePiece":
        #     # self.model.entity_representations[0].base[0].save_assignment(output_path=Path(path))
        #     relation_embedd = self.model.relation_representations[0].base._embeddings
        #     entity_embedd = (
        #         self.model.entity_representations[0].base[0].vocabulary._embeddings
        #     )
        #     return (
        #         entity_embedd.weight.data.detach(),
        #         relation_embedd.weight.data.detach(),
        #     )
        # if hasattr(self.model, "base"):
        #     if len(self.model.base.relation_representations) != 0:
        #         relation_embedd = self.model.base.relation_representations[
        #             0
        #         ]._embeddings
        #     if len(self.model.base.entity_representations) != 0:
        #         entity_embedd = self.model.base.entity_representations[0]._embeddings

        if (
            hasattr(self.model, "relation_representations")
            and len(self.model.relation_representations) != 0
        ):
            # TODO: number of emebedding index will be increased to twice as many as other model
            # problem need to be solved, otherwise it cannot be save to pd.DataFrame properly!!!
            # if isinstance(
            #     self.model.relation_representations[0], SingleCompGCNRepresentation
            # ):
            #     relation_embedd = self.model.relation_representations[
            #         0
            #     ].combined.relation_representations._embeddings
            # else:
            #     relation_embedd = self.model.relation_representations[0]._embeddings
            relation_embedd = self.model.relation_representations[0]()

        if (
            hasattr(self.model, "entity_representations")
            and len(self.model.entity_representations) != 0
        ):

            # if isinstance(
            #     self.model.entity_representations[0], SingleCompGCNRepresentation
            # ):
            #     entity_embedd = self.model.entity_representations[
            #         0
            #     ].combined.entity_representations._embeddings
            # else:
            #     entity_embedd = self.model.entity_representations[0]._embeddings
            entity_embedd = self.model.entity_representations[0]()

        return (
            entity_embedd.data.detach() if entity_embedd != None else None,
            relation_embedd.data.detach() if relation_embedd != None else None,
        )

    def training_epoch_end(self, training_step_outputs) -> None:
        batch_losses = [i["loss"].item() for i in training_step_outputs]
        avg = sum(batch_losses) / len(batch_losses)
        self.loss_history.append(avg)

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # the tensors here is inference tensors and can't be modified in-place outside InferenceMode.
        # https://twitter.com/PyTorch/status/1437838242418671620?s=20&t=8pEheJu4kRaLyJHBBLUvZA (solution)
        return predict.predict_triples(model=self.model, triples=x,).scores.clone()

