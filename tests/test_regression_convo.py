from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestRegressionConvO:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = argparse_default([])
        args.model = 'ConvO'
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.test_mode = True
        args.eval = True
        args.eval_on_train = True
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.torch_trainer = 'DataParallelTrainer'
        result = Execute(args).start()
        assert 1.0 >= result['Train']['H@1'] >= 0.01
        assert 0.75 >= result['Val']['H@1'] >= 0.01
        assert 0.75 >= result['Test']['H@1'] >= 0.01


    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = argparse_default([])
        args.model = 'ConvO'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval = True
        args.eval_on_train = True
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.torch_trainer = 'DataParallelTrainer'
        args.scoring_technique = '1vsAll'
        result = Execute(args).start()
        assert 1.0 >= result['Train']['H@1'] >= 0.70
        assert 0.85 >= result['Val']['H@1'] >= 0.70
        assert 0.85 >= result['Test']['H@1'] >= 0.70

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = argparse_default([])
        args.model = 'ConvO'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.eval = 1
        args.eval_on_train = 1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.torch_trainer = 'DataParallelTrainer'
        result = Execute(args).start()
        assert 1.0 >= result['Train']['H@1'] >= 0.01
        assert 0.75 >= result['Val']['H@1'] >= 0.01
        assert 0.75 >= result['Test']['H@1'] >= 0.01

