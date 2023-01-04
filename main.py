from core.executer import Execute
import argparse
import pytorch_lightning as pl


def argparse_default(description=None):
    """ Extends pytorch_lightning Trainer's arguments with ours """
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser(add_help=False))
    # Default Trainer param https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#methods
    # Dataset and storage related
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/KINSHIP',
                        help="The path of a folder containing input data")
    parser.add_argument("--save_embeddings_as_csv", type=bool, default=False,
                        help='A flag for saving embeddings in csv file.')
    parser.add_argument("--storage_path", type=str, default='Experiments',
                        help="Embeddings, model, and any other related data will be stored therein.")
    # Model and Training Parameters
    parser.add_argument("--model", type=str,
                        default="AConEx",
                        help="Available models: ConEx, ConvQ, ConvO, DistMult, QMult, OMult, "
                             "Shallom, AConEx, ConEx, ComplEx, DistMult, TransE, CLf")
    parser.add_argument('--optim', type=str, default='Adam',
                        help='[Adan, NAdam, Adam, SGD, Sls, AdamSLS]')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Number of dimensions for an embedding vector. ')
    parser.add_argument("--num_epochs", type=int, default=200, help='Number of epochs for training. ')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini batch size')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument('--callbacks',
                        '--list',
                        nargs='+',
                        default=[],  # PPE10 ,PPE20 or PPE
                        help='List of tuples representing a callback and values')
    parser.add_argument("--backend", type=str, default='polars',
                        help='Select [polars(seperator: \t), modin(seperator: \s+), pandas(seperator: \s+)]')
    parser.add_argument("--trainer", type=str, default='torchDDP',
                        help='PL (pytorch lightning trainer), torchDDP (custom ddp), torchCPUTrainer (custom cpu only)')
    # Hyperparameters for training.
    parser.add_argument('--scoring_technique', default='KvsAll', help="KvsSample, 1vsAll, KvsAll, NegSample")
    parser.add_argument('--neg_ratio', type=int, default=0,
                        help='The number of negative triples generated per positive triple.')
    # Optimization related hyperparameters
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 penalty e.g.(0.00001)')
    parser.add_argument('--input_dropout_rate', type=float, default=0.0)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.0)
    parser.add_argument("--feature_map_dropout_rate", type=int, default=0.0)
    parser.add_argument("--normalization", type=str, default="LayerNorm", help="[LayerNorm, BatchNorm1d, None]")
    parser.add_argument("--init_param", type=str, default=None, help="[xavier_normal]")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=0,
                        help="e.g. gradient_accumulation_steps=2 implies that gradients are accumulated at every second mini-batch")

    # Flags for computation
    parser.add_argument('--num_folds_for_cv', type=int, default=0, help='Number of folds in k-fold cross validation.'
                                                                        'If >2 ,no evaluation scenario is applied implies no evaluation.')
    parser.add_argument("--eval_model", type=str, default="train_val_test",
                        help='train, val, test, constraint, combine them anyway you want, e.g. '
                             'train_val,train_val_test, val_test, val_test_constraint ')
    # Additional training params
    parser.add_argument("--save_model_at_every_epoch", type=int, default=None,
                        help='At every X number of epochs model will be saved. If None, we save 4 times.')
    parser.add_argument("--label_smoothing_rate", type=float, default=0.0, help='None for not using it.')

    parser.add_argument("--kernel_size", type=int, default=3, help="Square kernel size for ConEx")
    parser.add_argument("--num_of_output_channels", type=int, default=32, help="# of output channels in convolution")

    parser.add_argument("--num_core", type=int, default=1, help='Number of cores to be used. 0 implies using single CPU')
    parser.add_argument("--seed_for_computation", type=int, default=0, help='Seed for all, see pl seed_everything().')
    parser.add_argument("--sample_triples_ratio", type=float, default=None, help='Sample input data.')
    parser.add_argument("--read_only_few", type=int, default=None, help='READ only first N triples. If 0, read all.')
    # @TODO: Do we still need it
    parser.add_argument("--min_freq_for_vocab", type=int, default=None,
                        help='Min number of triples for a vocab term to be considered')

    if description is None:
        return parser.parse_args()
    return parser.parse_args(description)


if __name__ == '__main__':
    report = Execute(argparse_default()).start()
    """
    {'num_train_triples': .., 'num_entities': .., 'num_relations': .., 
    'Train': {'H@1': .., 'H@3': .., 'H@10': .., 'MRR': ..}, 
    'Val': {'H@1': .., 'H@3': .., 'H@10': .., 'MRR': ..},  
    'Test': {'H@1': .., 'H@3': .., 'H@10': .., 'MRR': ..},  
    'NumParam': .., 'EstimatedSizeMB': .., 'Runtime': '.., 
    'path_experiment_folder': ..}
    """
