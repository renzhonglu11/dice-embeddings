from main import argparse_default
from core.executer import Execute
import sys
import pytest
import GPUtil

class TestCallback:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex_torch_cpu_trainer(self):
        args = argparse_default([])
        args.model = 'AConEx'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_aconex_pl_trainer(self):
        args = argparse_default([])
        args.model = 'AConEx'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.trainer = 'PL'
        Execute(args).start()


    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_aconex_torchDDP_trainer(self):
        args = argparse_default([])
        args.model = 'AConEx'
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 32
        args.lr = 0.01
        args.embedding_dim = 32
        args.trainer = 'torchDDP'
        args.num_core = 4 # need to be bigger than 0
        args.eval_model = 'train_val_test'
        Execute(args).start()



