import json
import logging
import time
import warnings
from types import SimpleNamespace
import os
import datetime
import argparse
import numpy as np
from pytorch_lightning import seed_everything

from dicee.knowledge_graph import KG
from dicee.models.base_model import BaseKGE
from dicee.evaluator import Evaluator
# Avoid
from dicee.static_funcs import *
from dicee.static_preprocess_funcs import preprocesses_input_args
from dicee.sanity_checkers import *
from dicee.trainer import DICE_Trainer
import pytorch_lightning as pl

logging.getLogger('pytorch_lightning').setLevel(0)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


class Execute:
    """ A class for Training, Retraining and Evaluation a model.

    (1) Loading & Preprocessing & Serializing input data.
    (2) Training & Validation & Testing
    (3) Storing all necessary info
    """

    def __init__(self, args, continuous_training=False):
        # (1) Process arguments and sanity checking.
        self.args = preprocesses_input_args(args)
        # (2) Ensure reproducibility.
        seed_everything(args.seed_for_computation, workers=True)
        # (3) Set the continual training flag
        self.is_continual_training = continuous_training
        # (4) Create an experiment folder or use the previous one
        continual_training_setup_executor(self)
        # (5) A variable is initialized for pytorch lightning trainer or DICE_Trainer()
        self.trainer = None
        self.trained_model = None
        # (6) A variable is initialized for storing input data.
        self.dataset = None
        # (7) Store few data in memory for numerical results, e.g. runtime, H@1 etc.
        self.report = dict()
        # (8) Create an object to carry out link prediction evaluations
        self.evaluator = None  # e.g. Evaluator(self)

    def read_preprocess_index_serialize_data(self) -> None:
        """ Read & Preprocess & Index & Serialize Input Data

        (1) Read or load the data from disk into memory.
        (2) Store the statistics of the data.

        Parameter
        ----------

        Return
        ----------
        None

        """
        # (1) Read & Preprocess & Index & Serialize Input Data.
        self.dataset = read_or_load_kg(self.args, cls=KG)
        # (2) Sanity checking.
        self.args, self.dataset = config_kge_sanity_checking(self.args, self.dataset)
        # (3) Store the stats
        self.args.num_entities = self.dataset.num_entities
        self.args.num_relations = self.dataset.num_relations
        self.report['num_train_triples'] = len(self.dataset.train_set)
        self.report['num_entities'] = self.dataset.num_entities
        self.report['num_relations'] = self.dataset.num_relations


    
    def load_indexed_data(self) -> None:
        """ Load the indexed data from disk into memory

        Parameter
        ----------

        Return
        ----------
        None

        """
        self.dataset = read_or_load_kg(self.args, cls=KG)

    @timeit
    def save_trained_model(self) -> None:
        """ Save a knowledge graph embedding model

        (1) Send model to eval mode and cpu.
        (2) Store the memory footprint of the model.
        (3) Save the model into disk.
        (4) Update the stats of KG again ?

        Parameter
        ----------

        Return
        ----------
        None

        """
        print('*** Save Trained Model ***')
        self.trained_model.eval()
        self.trained_model.to('cpu')
        # Save the epoch loss
        # (2) Store NumParam and EstimatedSizeMB
        self.report.update(self.trained_model.mem_of_model())
        # (3) Store/Serialize Model for further use.
        if self.is_continual_training is False:
            store(trainer=self.trainer,
                  trained_model=self.trained_model,
                  model_name='model',
                  full_storage_path=self.storage_path,
                  save_as_csv=self.args.save_embeddings_as_csv)
        else:
            store(trainer=self.trainer,
                  trained_model=self.trained_model,
                  model_name='model_' + str(datetime.datetime.now()),
                  full_storage_path=self.storage_path, save_as_csv=self.args.save_embeddings_as_csv)

        self.report['path_experiment_folder'] = self.storage_path
        self.report['num_entities'] = self.args.num_entities
        self.report['num_relations'] = self.args.num_relations
        self.report['path_experiment_folder'] = self.storage_path

    def end(self, start_time, form_of_labelling) -> dict:
        """
        End training

        (1) Store trained model.
        (2) Report runtimes.
        (3) Eval model if required.

        Parameter
        ---------

        Returns
        -------
        A dict containing information about the training and/or evaluation

        """
        # (1) Save the model
        self.save_trained_model()
        # (2) Update and inform the runtime
        self.report['Runtime'] = time.time() - start_time
        print(f"Total computation time: {self.report['Runtime']:.3f} seconds")
        # (3) Store the report of training.
        with open(self.args.full_storage_path + '/report.json', 'w') as file_descriptor:
            json.dump(self.report, file_descriptor, indent=4)
        # (4) Eval model and return eval results.
        if self.args.eval_model is None:
            return {**self.report}
        else:
            self.evaluator.eval(dataset=self.dataset, trained_model=self.trained_model,
                                form_of_labelling=form_of_labelling)
            return {**self.report, **self.evaluator.report}

    def start(self) -> dict:
        """
        Start training

        # (1) Loading the Data
        # (2) Create an evaluator object.
        # (3) Create a trainer object.
        # (4) Start the training

        Parameter
        ---------

        Returns
        -------
        A dict containing information about the training and/or evaluation

        """
    #     config = {
    #     "epoch":self.args.num_epochs,"lr":self.args.lr,"embedding_dim":self.args.embedding_dim,"optimizer":self.args.optim
    # }
    #     dataset = self.args.path_dataset_folder.split('/')[1]
    #     wandb.init(project="dice_demo",config=config,name=f'{self.args.model}-{dataset}')

        start_time = time.time()
        print(f"Start time:{datetime.datetime.now()}")
        # (1) Loading the Data
        #  Load the indexed data from disk or read a raw data from disk.
        self.load_indexed_data() if self.is_continual_training else self.read_preprocess_index_serialize_data()
        # (2) Create an evaluator object.
        self.evaluator = Evaluator(args=self.args)
        # (3) Create a trainer object.
        self.trainer = DICE_Trainer(args=self.args,
                                    is_continual_training=self.is_continual_training,
                                    storage_path=self.storage_path,
                                    evaluator=self.evaluator)
        # (4) Start the training
        self.trained_model, form_of_labelling = self.trainer.start(dataset=self.dataset)
        return self.end(start_time, form_of_labelling)


class ContinuousExecute(Execute):
    """ A subclass of Execute Class for retraining

    (1) Loading & Preprocessing & Serializing input data.
    (2) Training & Validation & Testing
    (3) Storing all necessary info
    """

    def __init__(self, args):
        assert os.path.exists(args.path_experiment_folder)
        assert os.path.isfile(args.path_experiment_folder + '/configuration.json')
        # (1) Load Previous input configuration
        previous_args = load_json(args.path_experiment_folder + '/configuration.json')
        dargs = vars(args)
        del args
        for k in list(dargs.keys()):
            if dargs[k] is None:
                del dargs[k]
        # (2) Update (1) with new input
        previous_args.update(dargs)
        try:
            report = load_json(dargs['path_experiment_folder'] + '/report.json')
            previous_args['num_entities'] = report['num_entities']
            previous_args['num_relations'] = report['num_relations']
        except AssertionError:
            print("Couldn't find report.json.")
        previous_args = SimpleNamespace(**previous_args)
        previous_args.full_storage_path = previous_args.path_experiment_folder
        print('ContinuousExecute starting...')
        print(previous_args)
        # TODO: can we remove continuous_training from Execute ?
        super().__init__(previous_args, continuous_training=True)

    def continual_start(self) -> dict:
        """
        Start Continual Training

        (1) Initialize training.
        (2) Start continual training.
        (3) Save trained model.

        Parameter
        ---------

        Returns
        -------
        A dict containing information about the training and/or evaluation

        """
        # (1)
        self.trainer = DICE_Trainer(args=self.args, is_continual_training=True,
                                    storage_path=self.args.path_experiment_folder)
        # (2)
        self.trained_model, form_of_labelling = self.trainer.continual_start()

        # (5) Store trained model.
        self.save_trained_model()
        # (6) Eval model.
        if self.args.eval_model is None:
            return self.report
        else:
            self.evaluator = Evaluator(args=self.args, is_continual_training=True)
            self.evaluator.dummy_eval(self.trained_model, form_of_labelling)
            return {**self.report, **self.evaluator.report}

class ParseDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict()) # set each name of the attribute to hold the created object(s) as dictionary
        for value in values:
            key, value = value.split('=')
            if value.isdigit():
                getattr(namespace, self.dest)[key] = int(value)
                continue
            getattr(namespace, self.dest)[key] = value


def get_default_arguments(description=None):
    """ Extends pytorch_lightning Trainer's arguments with ours """
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser(add_help=False))
    # Default Trainer param https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#methods
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS',
                        help="The path of a folder containing input data")
    parser.add_argument("--save_embeddings_as_csv", type=bool, default=False,
                        help='A flag for saving embeddings in csv file.')
    parser.add_argument("--storage_path", type=str, default='Experiments',
                        help="Embeddings, model, and any other related data will be stored therein.")
    parser.add_argument("--model", type=str,
                        default="Keci",
                        help="Available models: CMult, ConEx, ConvQ, ConvO, DistMult, QMult, OMult, "
                             "Shallom, AConEx, ConEx, ComplEx, DistMult, TransE, CLf")
    parser.add_argument('--p', type=int, default=0,
                        help='P for Clifford Algebra')
    parser.add_argument('--q', type=int, default=0,
                        help='Q for Clifford Algebra')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='[Adan, NAdam, Adam, SGD, Sls, AdamSLS]')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Number of dimensions for an embedding vector. ')
    parser.add_argument("--num_epochs", type=int, default=10, help='Number of epochs for training. ')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini batch size')
    parser.add_argument('--auto_batch_finder', type=bool, default=False,
                        help='Find a batch size w.r.t. computational budgets')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument('--callbacks', '--list', nargs='+', default=[],
                        help='List of tuples representing a callback and values, e.g. [FPPE or PPE or PPE10 ,PPE20 or PPE, FPPE]')
    parser.add_argument("--backend", type=str, default='pandas',
                        help='Select [polars(seperator: \t), modin(seperator: \s+), pandas(seperator: \s+)]')
    parser.add_argument("--trainer", type=str, default='torchCPUTrainer',
                        help='PL (pytorch lightning trainer), torchDDP (custom ddp), torchCPUTrainer (custom cpu only)')
    parser.add_argument('--scoring_technique', default='KvsAll', help="KvsSample, 1vsAll, KvsAll, NegSample")
    parser.add_argument('--neg_ratio', type=int, default=0,
                        help='The number of negative triples generated per positive triple.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 penalty e.g.(0.00001)')
    parser.add_argument('--input_dropout_rate', type=float, default=0.0)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.0)
    parser.add_argument("--feature_map_dropout_rate", type=int, default=0.0)
    parser.add_argument("--normalization", type=str, default="None", help="[LayerNorm, BatchNorm1d, None]")
    parser.add_argument("--init_param", type=str, default=None, help="[xavier_normal, None]")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0,
                        help="e.g. gradient_accumulation_steps=2 implies that gradients are accumulated at every second mini-batch")
    parser.add_argument('--num_folds_for_cv', type=int, default=0,
                        help='Number of folds in k-fold cross validation.'
                             'If >2 ,no evaluation scenario is applied implies no evaluation.')
    parser.add_argument("--eval_model", type=str, default="train_val_test",
                        help='train, val, test, constraint, combine them anyway you want, e.g. '
                             'train_val,train_val_test, val_test, val_test_constraint ')
    parser.add_argument("--save_model_at_every_epoch", type=int, default=None,
                        help='At every X number of epochs model will be saved. If None, we save 4 times.')
    parser.add_argument("--label_smoothing_rate", type=float, default=0.0, help='None for not using it.')
    parser.add_argument("--kernel_size", type=int, default=3, help="Square kernel size for ConEx")
    parser.add_argument("--num_of_output_channels", type=int, default=32,
                        help="# of output channels in convolution")
    parser.add_argument("--num_core", type=int, default=0,
                        help='Number of cores to be used. 0 implies using single CPU')
    parser.add_argument("--seed_for_computation", type=int, default=0,
                        help='Seed for all, see pl seed_everything().')
    parser.add_argument("--sample_triples_ratio", type=float, default=None, help='Sample input data.')
    parser.add_argument("--read_only_few", type=int, default=None,
                        help='READ only first N triples. If 0, read all.')
    parser.add_argument("--pykeen_model_kwargs", nargs='*',action=ParseDict, help='addtional paramters pass to pykeen_model')
    parser.add_argument("--use_SLCWALitModule",action="store_true",help='whether to use SLCWALitModule in pykeen or not')
    parser.add_argument("--use_ddp_batch_finder", action="store_true", help="whether to use batch finder in ddp" )
    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)
