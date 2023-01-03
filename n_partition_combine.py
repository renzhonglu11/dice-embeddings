import itertools
from argparse import ArgumentParser
from glob import glob
import torch
from core.executer import Execute
from core import load_json
from core.static_funcs import load_model, intialize_model, create_experiment_folder
import pandas as pd
import numpy as np
import os
import json


class Merger:
    def __init__(self, args):
        self.argument_sanity_checking(args)
        self.args = args
        # 2 Create a folder to serialize data and replace the previous path info
        self.args.full_storage_path = create_experiment_folder(folder_name=self.args.storage_path)
        self.configuration = None
        self.model_name = None

        self.entity_embeddings = None
        self.relation_embeddings = None
        self.embedding_dim = None
        self.merged_pre_trained_model = None

    def model_type_sanity_checking(self, name, previous_args):
        if self.model_name is None:
            self.model_name = name
        else:
            try:
                assert self.model_name == name
            except AssertionError:
                raise AssertionError(f'{self.model_name} can be ensembled with {previous_args["model"]}')

        self.configuration = previous_args

    def load_entity_embeddings(self, previous_args):

        path_entity_emb = previous_args['full_storage_path'] + f'/{self.model_name}_entity_embeddings'

        if os.path.isfile(path_entity_emb + '.csv'):
            df_entities = pd.read_csv(path_entity_emb + '.csv', index_col=0)
        elif os.path.isfile(path_entity_emb + '.npz'):

            df_entities = pd.DataFrame(data=np.load(path_entity_emb + '.npz')['entity_emb'],
                                       index=pd.read_parquet(
                                           path=previous_args[
                                                    'full_storage_path'] + f'/entity_to_idx.gzip').index)

        elif os.path.isfile(path_entity_emb + '.pt'):
            df_entities = pd.DataFrame(data=torch.load(path_entity_emb + '.pt', torch.device('cpu')).numpy(),
                                       index=pd.read_parquet(
                                           path=previous_args[
                                                    'full_storage_path'] + f'/entity_to_idx.gzip').index)
        else:
            raise FileNotFoundError(
                f"{previous_args['full_storage_path']} + f'/{self.model_name}_entity_embeddings")
        return df_entities

    def load_relation_embeddings(self, previous_args):

        path_relations_emb = previous_args['full_storage_path'] + f'/{self.model_name}_relation_embeddings'

        if os.path.isfile(path_relations_emb + '.csv'):
            df_relations = pd.read_csv(path_relations_emb + '.csv', index_col=0)
        elif os.path.isfile(path_relations_emb + '.npz'):
            df_relations = pd.DataFrame(data=np.load(path_relations_emb + '.npz')['relation_ebm'],
                                        index=pd.read_parquet(
                                            path=previous_args[
                                                     'full_storage_path'] + f'/relation_to_idx.gzip').index)
        elif os.path.isfile(path_relations_emb + '.pt'):
            df_relations = pd.DataFrame(data=torch.load(path_relations_emb + '.pt', torch.device('cpu')).numpy(),
                                        index=pd.read_parquet(
                                            path=previous_args[
                                                     'full_storage_path'] + f'/relation_to_idx.gzip').index)
        else:
            raise FileNotFoundError(
                f"{previous_args['full_storage_path']} + f'/{self.model_name}_relation_embeddings")
        return df_relations

    def load_embedding_matrices(self):
        """ Accumulate embedding matrices """
        # (1) Temporarily initialize embeddings as an empty list
        self.entity_embeddings = []
        self.relation_embeddings = []
        # (2) Counters for sanity checking in the concatenated dataframes.
        num_entity_rows = 0
        num_relation_rows = 0
        print('Loading models..')
        for path_experiment_folder in self.args.trained_model_paths:
            previous_args = load_json(path_experiment_folder + '/configuration.json')

            self.model_type_sanity_checking(previous_args['model'], previous_args)

            df_entities = self.load_entity_embeddings(previous_args)
            df_relations = self.load_relation_embeddings(previous_args)
            num_entity_rows += len(df_entities)
            num_relation_rows += len(df_relations)
            self.entity_embeddings.append(df_entities)
            self.relation_embeddings.append(df_relations)

        # (2) Concatenate entity embedding dataframes
        self.entity_embeddings = pd.concat(self.entity_embeddings, ignore_index=False)
        self.entity_embeddings.columns = self.entity_embeddings.columns.astype(str)
        assert len(self.entity_embeddings) == num_entity_rows
        self.relation_embeddings = pd.concat(self.relation_embeddings, ignore_index=False)
        self.relation_embeddings.columns = self.relation_embeddings.columns.astype(str)
        assert len(self.relation_embeddings) == num_relation_rows

        # Average overlapping indexes.
        self.entity_embeddings = self.entity_embeddings.groupby(self.entity_embeddings.index).mean()
        self.relation_embeddings = self.relation_embeddings.groupby(self.relation_embeddings.index).mean()
        # Write into disk
        self.entity_embeddings.to_parquet(self.args.full_storage_path + f'/{self.model_name}_entity_embeddings',
                                          compression='gzip', engine='pyarrow')
        pd.DataFrame(data=np.arange(len(self.entity_embeddings)),
                     columns=['entity'],
                     index=self.entity_embeddings.index).to_parquet(self.args.full_storage_path + '/entity_to_idx',
                                                                    compression='gzip', engine='pyarrow')

        self.relation_embeddings.to_parquet(
            self.args.full_storage_path + f'/{self.model_name}_relation_embeddings', compression='gzip',
            engine='pyarrow')

        pd.DataFrame(data=np.arange(len(self.relation_embeddings)),
                     columns=['relation'],
                     index=self.relation_embeddings.index).to_parquet(
            self.args.full_storage_path + '/relation_to_idx', compression='gzip', engine='pyarrow')

    def weights_averaging(self):
        """

        :return:
        """
        print('Averaging weights..')
        # (1) Average embeddings of entities sharing same index.

        # (2) Average non embedding weights
        non_embedding_weights_state_dict = None
        num_models = 0
        for path_experiment_folder in self.args.trained_model_paths:
            previous_args = load_json(path_experiment_folder + '/configuration.json')
            previous_report = load_json(path_experiment_folder + '/report.json')

            previous_args['path_of_experiment_folder'] = previous_args['full_storage_path']
            if self.model_name is None:
                self.model_name = previous_args['model']
            else:
                try:
                    assert self.model_name == previous_args['model']
                except AssertionError:
                    raise AssertionError(f'{self.model_name} can be ensembled with {previous_args["model"]}')
            num_models += 1.

            previous_args['num_entities'] = previous_report['num_entities']
            previous_args['num_relations'] = previous_report['num_relations']

            i_th_pretrained, _, __ = load_model(path_of_experiment_folder=previous_args['path_of_experiment_folder'])
            i_th_state_dict = i_th_pretrained.state_dict()

            if non_embedding_weights_state_dict is None:
                non_embedding_weights_state_dict = i_th_state_dict
            else:
                for key in i_th_state_dict:
                    if 'emb' not in key:
                        # Batch norm has num_batches_tracked param. I dunno what do do with it iat merging
                        if non_embedding_weights_state_dict[key].dtype == torch.float:
                            non_embedding_weights_state_dict[key] += i_th_state_dict[key]

        embdding_keys = []
        for key in non_embedding_weights_state_dict:
            if 'emb' not in key:
                # Batch norm has num_batches_tracked param. I dunno what do do with it iat merging
                if non_embedding_weights_state_dict[key].dtype == torch.float:
                    print('Updated:', key)
                    non_embedding_weights_state_dict[key] /= num_models
            else:
                embdding_keys.append(key)
        print(embdding_keys)
        exit(1)
        embedding_entity_keys, embedding_relation_keys = embdding_keys[:len(embdding_keys) // 2], embdding_keys[
                                                                                                  len(embdding_keys) // 2:]

        assert len(embedding_entity_keys) == len(embedding_relation_keys)
        for key, val in zip(embedding_entity_keys, torch.hsplit(torch.from_numpy(self.entity_embeddings.to_numpy()),
                                                                len(embedding_entity_keys))):
            non_embedding_weights_state_dict[key] = val

        for key, val in zip(embedding_relation_keys, torch.hsplit(torch.from_numpy(self.relation_embeddings.to_numpy()),
                                                                  len(embedding_relation_keys))):
            non_embedding_weights_state_dict[key] = val

        self.configuration['num_entities'] = len(self.entity_embeddings)
        self.configuration['num_relations'] = len(self.relation_embeddings)
        self.configuration['model'] = self.model_name

        self.args.num_entities = len(self.entity_embeddings)
        self.args.num_relations = len(self.relation_embeddings)

        self.args.model = self.model_name
        self.args.embedding_dim = self.entity_embeddings.shape[1] // len(embedding_entity_keys)
        self.args.num_relations = self.relation_embeddings.shape[0]
        self.args.num_entities = self.entity_embeddings.shape[0]

        print(non_embedding_weights_state_dict)
        exit(1)
        final_model, _ = intialize_model(self.configuration)
        final_model.load_state_dict(non_embedding_weights_state_dict)
        final_model.eval()

        print('Storing merged model..')
        store_kge(final_model, path=self.args.full_storage_path + f'/model.pt')

        with open(self.args.full_storage_path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            json.dump(temp, file_descriptor)

        with open(self.args.full_storage_path + '/report.json', 'w') as file_descriptor:
            temp = vars(self.args)
            json.dump(temp, file_descriptor)

        return final_model

    def start(self):
        # (1) Load & Concatenate embedding dataframes
        self.load_embedding_matrices()
        # (2) Take the average of duplicated indexes
        self.merged_pre_trained_model = self.weights_averaging()
        print('Done!')

    @staticmethod
    def argument_sanity_checking(args):
        # Sanity checking

        for path_experiment_folder in args.trained_model_paths:
            assert os.path.exists(path_experiment_folder)
            assert os.path.isfile(path_experiment_folder + '/idx_train_df.gzip')
            assert os.path.isfile(path_experiment_folder + '/configuration.json')
        else:
            args.trained_model_paths = [i for i in glob("Experiments/*", recursive=False)]


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--storage_path", type=str, default='Merged')
    parser.add_argument("--trained_model_paths",
                        nargs="*",  # 0 or more values expected => creates a list
                        type=str,
                        default=[],  # default if nothing is provided
                        # default=[],  # If empty, work on all data
                        )
    raise NotImplementedError()
    Merger(parser.parse_args()).start()
