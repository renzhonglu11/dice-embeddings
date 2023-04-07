from dicee.executer import Execute, get_default_arguments
import sys
import pytest


class TestDefaultParams:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_shallom(self):
        args = get_default_arguments([])
        args.model = 'Shallom'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex(self):
        args = get_default_arguments([])
        args.model = 'ConEx'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult(self):
        args = get_default_arguments([])
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.trainer = "PL"
        args.eval_model = 'train_val_test'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convq(self):
        args = get_default_arguments([])
        args.model = 'ConvQ'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_omult(self):
        args = get_default_arguments([])
        args.model = 'OMult'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.trainer = None
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convo(self):
        args = get_default_arguments([])
        args.model = 'ConvO'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    def test_distmult(self):
        args = get_default_arguments([])
        args.model = 'DistMult'
        # args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.embedding_dim = 64
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        # args.trainer = 'torchCPUTrainer'
        args.trainer = 'PL'
        args.neg_ratio = 1
        args.eval_model = 'train_val_test'
        args.num_core = 0
        args.save_embeddings_as_csv = True
        args.normalization = None
        args.optim = 'Adam'
        
        import wandb
        config = {
        "epoch":args.num_epochs,"lr":args.lr,"embedding_dim":args.embedding_dim
    }
        dataset = args.path_dataset_folder.split('/')[1]
        wandb.setup(wandb.Settings(program="test_pykeen_model.py", program_relpath="test_pykeen_model.py"))
        wandb.init(project="dice_demo",config=config,name=f'{args.model}-{dataset}')
        
        Execute(args).start()


