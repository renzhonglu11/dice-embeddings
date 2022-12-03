from main import argparse_default
from core.executer import Execute
import sys
import pytest


def template(model_name):
    args = argparse_default([])
    args.model = model_name
    args.scoring_technique = "NegSample"
    args.path_dataset_folder = "KGs/Nations"
    args.num_epochs = 10
    args.batch_size = 1024
    args.lr = 0.01
    args.embedding_dim = 50
    args.input_dropout_rate = 0.0
    args.hidden_dropout_rate = 0.0
    args.feature_map_dropout_rate = 0.0
    args.sample_triples_ratio = None
    args.read_only_few = None
    args.sample_triples_ratio = None
    args.torch_trainer = "None"
    args.neg_ratio = 1
    args.pykeen_model_kwargs = dict(
        embedding_dim=args.embedding_dim, loss="bcewithlogits",
    )
    args.use_SLCWALitModule = False
    Execute(args).start()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "model_name",
    [
        "Pykeen_DistMult",
        "Pykeen_TuckER",
        "Pykeen_UM",
        "Pykeen_TransR",
        "Pykeen_TransH",
        "Pykeen_TransF",
        "Pykeen_TransE",
        "Pykeen_TransD",
        "Pykeen_TorusE",
        "Pykeen_SimplE",
        "Pykeen_SE",
        "Pykeen_RESCAL",
        "Pykeen_RotatE",
        "Pykeen_QuatE",
        "Pykeen_PairRE",
        "Pykeen_ProjE",
        "Pykeen_NTN",
        "Pykeen_NodePiece",
        "Pykeen_MuRE",
        "Pykeen_KG2E",
        "Pykeen_InductiveNodePiece",
        "Pykeen_InductiveNodePieceGNN",
        "Pykeen_HolE",
        "Pykeen_FixedModel",
        "Pykeen_ERMLPE",
        "Pykeen_DistMA",
        "Pykeen_CrossE",
        "Pykeen_CooccurrenceFilteredModel",
        "Pykeen_ConvKB",
        "Pykeen_ConvE",
        "Pykeen_ComplExLiteral",
        "Pykeen_ComplEx",
        "Pykeen_CompGCN",
        "Pykeen_CP",
        "Pykeen_BoxE",
        "Pykeen_AutoSF",
        "Pykeen_DistMultLiteral",
    ],
)
def test_fixedModel(model_name):
    template(model_name)


class TestPykeen:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_specific_model(self):
        args = argparse_default([])
        args.model = "Pykeen_NodePiece"
        args.scoring_technique = "NegSample"
        args.path_dataset_folder = "KGs/Nations"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 50
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.torch_trainer = "None"
        args.neg_ratio = 1
        # args.use_SLCWALitModule = False
        args.pykeen_model_kwargs = dict(
            embedding_dim=args.embedding_dim,
            loss="bcewithlogits",
            # entity_representations=[None],
            tokenizers=["AnchorTokenizer", "RelationTokenizer"],
            num_tokens=[3, 12],
        )
        Execute(args).start()


# class TestPykeenLiteralModel:
#     @pytest.mark.filterwarnings('ignore::UserWarning')
#     def test_specific_model(self):
#         # "Pykeen_DistMultLiteralGated", "Pykeen_DistMultLiteral"
#         # literalModel in Pykeen has combinedRepresentation. That means two entity_representations will be returned
#         args = argparse_default([])
#         args.model = "Pykeen_DistMultLiteral"
#         args.num_epochs = 1
#         args.scoring_technique = "NegSample"
#         args.path_dataset_folder = "KGs/Nations"
#         args.num_epochs = 10
#         args.batch_size = 1024
#         args.lr = 0.01
#         args.embedding_dim = 50
#         args.input_dropout_rate = 0.0
#         args.hidden_dropout_rate = 0.0
#         args.feature_map_dropout_rate = 0.0
#         args.sample_triples_ratio = None
#         args.read_only_few = None
#         args.sample_triples_ratio = None
#         args.torch_trainer = "None"
#         args.neg_ratio = 1
#         args.pykeen_model_kwargs = dict(
#             embedding_dim=args.embedding_dim,
#             loss="bcewithlogits",
#         )
#         Execute(args).start()
