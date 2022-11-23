import time
from typing import Dict, List, Tuple, Union
from collections import defaultdict
import numpy as np
import pickle
import json
import os
import pandas
import pandas as pd
from .static_funcs import performance_debugger, get_er_vocab, get_ee_vocab, get_re_vocab, \
    create_recipriocal_triples, add_noisy_triples, index_triples, load_data_parallel, create_constraints, \
    numpy_data_type_changer, vocab_to_parquet, preprocess_dataframe_of_kg, dask_remove_triples_with_condition
from .sanity_checkers import dataset_sanity_checking
import glob
import pyarrow.parquet as pq


class KG:
    """ Knowledge Graph Class
        1- Reading : Large input data is read via DASK
        2- Cleaning & Preprocessing :
                                    Remove triples with literals if exists
                                    Apply reciprocal data augmentation triples into train, valid and test datasets
                                    Add noisy triples (random facts sampled from all possible triples E x R x E)
        3- Serializing and Deserializing in parquet format
    """

    def __init__(self, data_dir: str = None, deserialize_flag: str = None,
                 num_core: int = 1,
                 add_reciprical: bool = None, eval_model: str = None,
                 read_only_few: int = None, sample_triples_ratio: float = None,
                 path_for_serialization: str = None, add_noise_rate: float = None,
                 min_freq_for_vocab: int = None,
                 entity_to_idx=None, relation_to_idx=None, dnf_predicates=None, backend=None):
        """

        :param data_dir: A path of a folder containing the input knowledge graph
        :param deserialize_flag: A path of a folder containing previously parsed data
        :param large_kg_parse: A flag for using all cores to parse input knowledge graph
        :param add_reciprical: A flag for applying reciprocal data augmentation technique
        :param eval_model: A flag indicating whether evaluation will be applied. If no eval, then entity relation mappings will be deleted to free memory.
        :param add_noise_rate: Add say 10% noise in the input data
        sample_triples_ratio
        """
        self.num_entities = None
        self.num_relations = None
        self.df_str_kg = None
        self.data_dir = data_dir
        self.deserialize_flag = deserialize_flag
        self.num_core = num_core
        self.add_reciprical = add_reciprical
        self.eval_model = eval_model

        self.read_only_few = read_only_few
        self.sample_triples_ratio = sample_triples_ratio
        self.path_for_serialization = path_for_serialization
        self.add_noise_rate = add_noise_rate

        self.min_freq_for_vocab = min_freq_for_vocab
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.backend = 'pandas' if backend is None else backend
        self.input_is_parquet = None
        self.dnf_predicates = dnf_predicates

        # (1) Load + Preprocess input data.
        if deserialize_flag is None:
            # (1.1) Load and Preprocess the data.
            self.train_set, self.valid_set, self.test_set = self.load_read_process()
            # (1.2) Update (1.1).
            self.apply_reciprical_or_noise()
            # (1.3) Construct integer indexing for entities and relations.
            if entity_to_idx is None and relation_to_idx is None:
                self.sequential_vocabulary_construction()
                print('[9 / 14] Obtaining entity to integer index mapping from pandas dataframe...')
                # CD: <> brackets are not removed anymore <http://embedding.cc/resource/Karim_Fegrouch>
                self.entity_to_idx = self.entity_to_idx.to_dict()['entity']
                print('Done !\n')
                print('[9 / 14] Obtaining relation to integer index mapping from pandas dataframe...')
                self.relation_to_idx = self.relation_to_idx.to_dict()['relation']
                print('Done !\n')
                self.num_entities = len(self.entity_to_idx)
                self.num_relations = len(self.relation_to_idx)
                print('[10 / 14] Mapping training data into integers for training...')
                start_time = time.time()
                # 9. Use bijection mappings obtained in (4) and (5) to create training data for models.
                self.train_set = index_triples(self.train_set,
                                               self.entity_to_idx,
                                               self.relation_to_idx)
                print(f'Done ! {time.time() - start_time:.3f} seconds\n')
                if path_for_serialization is not None:
                    # 10. Serialize (9).
                    print('[11 / 14] Serializing integer mapped data...')
                    self.train_set.to_parquet(path_for_serialization + '/idx_train_df.gzip', compression='gzip',
                                              engine='pyarrow')
                    print('Done !\n')
                assert isinstance(self.train_set, pd.core.frame.DataFrame)
                # 11. Convert data from pandas dataframe to numpy ndarray.
                print('[12 / 14] Mapping from pandas data frame to numpy ndarray to reduce memory usage...')
                self.train_set = self.train_set.values
                print('Done !\n')
            else:
                print(
                    '[4 / 14] Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...',
                )
                self.entity_to_idx = self.entity_to_idx.to_dict()['entity']
                self.relation_to_idx = self.relation_to_idx.to_dict()['relation']
                self.num_entities = len(self.entity_to_idx)
                self.num_relations = len(self.relation_to_idx)
                print('Done !\n')
                print('[10 / 14] Mapping training data into integers for training...')
                # 9. Use bijection mappings obtained in (4) and (5) to create training data for models.
                self.train_set = index_triples(self.train_set, self.entity_to_idx, self.relation_to_idx)
                print('Done !\n')
                self.train_set = self.train_set
                assert isinstance(self.train_set, pd.core.frame.DataFrame)
                # 11. Convert data from pandas dataframe to numpy ndarray.
                print('[12 / 14] Mapping from pandas data frame to numpy ndarray to reduce memory usage...')
                self.train_set = self.train_set.values
                print('Done !\n')

            self.train_set = numpy_data_type_changer(self.train_set, num=max(self.num_entities, self.num_relations))
            print('[13 / 14 ] Sanity checking...')
            # 12. Sanity checking: indexed training set can not have an indexed entity assigned with larger indexed than the number of entities.
            dataset_sanity_checking(self.train_set, self.num_entities, self.num_relations)
            print('Done !\n')
            if self.valid_set is not None:
                if path_for_serialization is not None:
                    print('[14 / 14 ] Serializing validation data for Continual Learning...')
                    self.valid_set.to_parquet(
                        path_for_serialization + '/valid_df.gzip', compression='gzip', engine='pyarrow')
                    print('Done !\n')
                print('[14 / 14 ] Indexing validation dataset...')
                self.valid_set = index_triples(self.valid_set, self.entity_to_idx, self.relation_to_idx)
                print('Done !\n')
                if path_for_serialization is not None:
                    print('[15 / 14 ] Serializing indexed validation dataset...')
                    self.valid_set.to_parquet(
                        path_for_serialization + '/idx_valid_df.gzip', compression='gzip', engine='pyarrow')
                    print('Done !\n')
                # To numpy
                self.valid_set = self.valid_set.values
                dataset_sanity_checking(self.valid_set, self.num_entities, self.num_relations)
                self.valid_set = numpy_data_type_changer(self.valid_set, num=max(self.num_entities, self.num_relations))
            if self.test_set is not None:
                if path_for_serialization is not None:
                    print('[16 / 14 ] Serializing test data for Continual Learning...')
                    self.test_set.to_parquet(
                        path_for_serialization + '/test_df.gzip', compression='gzip', engine='pyarrow')
                    print('Done !\n')
                print('[17 / 14 ] Indexing test dataset...')
                self.test_set = index_triples(self.test_set, self.entity_to_idx, self.relation_to_idx)
                print('Done !\n')
                if path_for_serialization is not None:
                    print('[18 / 14 ] Serializing indexed test dataset...')
                    self.test_set.to_parquet(
                        path_for_serialization + '/idx_test_df.gzip', compression='gzip', engine='pyarrow')
                # To numpy
                self.test_set = self.test_set.values
                dataset_sanity_checking(self.test_set, self.num_entities, self.num_relations)
                self.test_set = numpy_data_type_changer(self.test_set, num=max(self.num_entities, self.num_relations))
                print('Done !\n')
            if eval_model:  # and len(self.valid_set) > 0 and len(self.test_set) > 0:
                if self.valid_set is not None and self.test_set is not None:
                    assert isinstance(self.valid_set, np.ndarray) and isinstance(self.test_set, np.ndarray)
                    # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                    data = np.concatenate([self.train_set, self.valid_set, self.test_set])
                else:
                    data = self.train_set
                print('Final: Creating Vocab...')
                self.er_vocab = get_er_vocab(data)
                self.re_vocab = get_re_vocab(data)
                # 17. Create a bijection mapping from subject-object pairs to relations.
                self.ee_vocab = get_ee_vocab(data)
                self.domain_constraints_per_rel, self.range_constraints_per_rel = create_constraints(self.train_set)
        else:
            self.deserialize(deserialize_flag)

            if self.eval_model:
                if self.valid_set is not None and self.test_set is not None:
                    # 16. Create a bijection mapping from subject-relation pairs to tail entities.
                    data = np.concatenate([self.train_set, self.valid_set, self.test_set])
                else:
                    data = self.train_set
                print('[7 / 4] Creating er,re, and ee type vocabulary for evaluation...')
                start_time = time.time()
                self.er_vocab = get_er_vocab(data)
                self.re_vocab = get_re_vocab(data)
                # 17. Create a bijection mapping from subject-object pairs to relations.
                self.ee_vocab = get_ee_vocab(data)
                self.domain_constraints_per_rel, self.range_constraints_per_rel = create_constraints(self.train_set)
                print(f'Done !\t{time.time() - start_time:.3f} seconds\n')

        # 4. Display info
        self.description_of_input = f'\n------------------- Description of Dataset {data_dir} -------------------'
        self.description_of_input += f'\nNumber of entities: {self.num_entities}' \
                                     f'\nNumber of relations: {self.num_relations}' \
                                     f'\nNumber of triples on train set: {len(self.train_set)}' \
                                     f'\nNumber of triples on valid set: {len(self.valid_set) if self.valid_set is not None else 0}' \
                                     f'\nNumber of triples on test set: {len(self.test_set) if self.test_set is not None else 0}\n'

    def sequential_vocabulary_construction(self) -> None:
        """
        (1) Read input data into memory
        (2) Remove triples with a condition
        (3) Serialize vocabularies in a pandas dataframe where
                    => the index is integer and
                    => a single column is string (e.g. URI)
        """
        try:
            assert isinstance(self.train_set, pd.DataFrame)
        except TypeError:
            print(type(self.train_set))
            print('HEREE')
            exit(1)
        assert isinstance(self.valid_set, pd.DataFrame) or self.valid_set is None
        assert isinstance(self.test_set, pd.DataFrame) or self.test_set is None

        # (4) Remove triples from (1).
        self.remove_triples_from_train_with_condition()
        # Concatenate dataframes.
        print('\n[4 / 14] Concatenating data to obtain index...')
        x = [self.train_set]
        if self.valid_set is not None:
            x.append(self.valid_set)
        if self.test_set is not None:
            x.append(self.test_set)
        df_str_kg = pd.concat(x, ignore_index=True)
        del x
        print('Done !\n')

        print('[5 / 14] Creating a mapping from entities to integer indexes...')
        # (5) Create a bijection mapping from entities of (2) to integer indexes.
        # ravel('K') => Return a contiguous flattened array.
        # ‘K’ means to read the elements in the order they occur in memory, except for reversing the data when strides are negative.
        ordered_list = pd.unique(df_str_kg[['subject', 'object']].values.ravel('K'))
        self.entity_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)), columns=['entity'], index=ordered_list)
        print('Done !\n')
        vocab_to_parquet(self.entity_to_idx, 'entity_to_idx.gzip', self.path_for_serialization,
                         print_into='[6 / 14] Serializing compressed entity integer mapping...')
        # 5. Create a bijection mapping  from relations to integer indexes.
        print('[7 / 14] Creating a mapping from relations to integer indexes...')
        ordered_list = pd.unique(df_str_kg['relation'].values.ravel('K'))
        self.relation_to_idx = pd.DataFrame(data=np.arange(len(ordered_list)),
                                            columns=['relation'],
                                            index=ordered_list)
        print('Done !\n')

        vocab_to_parquet(self.relation_to_idx, 'relation_to_idx.gzip', self.path_for_serialization,
                         '[8 / 14] Serializing compressed relation integer mapping...')
        del ordered_list

    def remove_triples_from_train_with_condition(self):
        # @TODO: Move to static_funcs.py
        if self.min_freq_for_vocab is not None:
            assert isinstance(self.min_freq_for_vocab, int)
            assert self.min_freq_for_vocab > 0
            print(
                f'[5 / 14] Dropping triples having infrequent entities or relations (>{self.min_freq_for_vocab})...',
                end=' ')
            num_triples = self.train_set.size
            print('Total num triples:', num_triples, end=' ')
            # Compute entity frequency: index is URI, val is number of occurrences.
            entity_frequency = pd.concat([self.train_set['subject'], self.train_set['object']]).value_counts()
            relation_frequency = self.train_set['relation'].value_counts()

            # low_frequency_entities index and values are the same URIs: dask.dataframe.core.DataFrame
            low_frequency_entities = entity_frequency[
                entity_frequency <= self.min_freq_for_vocab].index.values
            low_frequency_relation = relation_frequency[
                relation_frequency <= self.min_freq_for_vocab].index.values
            # If triple contains subject that is in low_freq, set False do not select
            self.train_set = self.train_set[~self.train_set['subject'].isin(low_frequency_entities)]
            # If triple contains object that is in low_freq, set False do not select
            self.train_set = self.train_set[~self.train_set['object'].isin(low_frequency_entities)]
            # If triple contains relation that is in low_freq, set False do not select
            self.train_set = self.train_set[~self.train_set['relation'].isin(low_frequency_relation)]
            # print('\t after dropping:', df_str_kg.size.compute(scheduler=scheduler_flag))
            print('\t after dropping:', self.train_set.size)  # .compute(scheduler=scheduler_flag))
            del low_frequency_entities
            print('Done !\n')

    def load_read_process(self) -> Tuple[pandas.core.frame.DataFrame, Union[pandas.core.frame.DataFrame, None], Union[
        pandas.core.frame.DataFrame, None]]:
        """Load and preprocess train, valid, and test datasets

        A data written in the disk (a csv, parquet or text format) is read into memory and preprocessed

        return:
        """
        # (1) Check whether a path leading to a directory is a parquet formatted file
        self.train_set, self.valid_set, self.test_set = None, None, None
        # (CD) Removed
        if self.dnf_predicates:
            # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
            # pq.read_table(self.data_dir,columns=['subject', 'relation', 'object'],filters=self.dnf_predicates)
            raise NotImplementedError

        for i in glob.glob(self.data_dir + '/*'):
            if 'train' in i:
                # 1. LOAD Data. (First pass on data)
                print(
                    f'[1 / 14] Loading and Preprocessing Training Data: read_only_few: {self.read_only_few} , sample_triples_ratio: {self.sample_triples_ratio}...')
                self.train_set = load_data_parallel(i, self.read_only_few, self.sample_triples_ratio,
                                                    backend=self.backend)
            elif 'test' in i:
                print(f'[1 / 14] Loading and Preprocessing Test Data:')
                self.test_set = load_data_parallel(i, backend=self.backend)
            elif 'valid' in i:
                print(f'[1 / 14] Loading and Preprocessing Valid Data:')
                self.valid_set = load_data_parallel(i, backend=self.backend)
            else:
                print(f'Unrecognized data {i}')

        """
        if self.data_dir[-8:] == '.parquet':
            print(
                f'[1 / 14] Read parquet formatted KG with pyarrow and preprocess: read_only_few: {self.read_only_few} , sample_triples_ratio: {self.sample_triples_ratio}...')
            if self.dnf_predicates:
                # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
                self.train_set = preprocess_dataframe_of_kg(df=pq.read_table(self.data_dir,
                                                                             columns=['subject', 'relation', 'object'],
                                                                             filters=self.dnf_predicates).to_pandas(),
                                                            read_only_few=self.read_only_few,
                                                            sample_triples_ratio=self.sample_triples_ratio)
            else:
                self.train_set = preprocess_dataframe_of_kg(df=pd.read_parquet(self.data_dir, engine='pyarrow'),
                                                            read_only_few=self.read_only_few,
                                                            sample_triples_ratio=self.sample_triples_ratio)
            print('Train Dataset:', self.train_set)
            print('Done !\n')
            self.valid_set = None
            self.test_set = None
            self.input_is_parquet = True
        else:
            # 1. LOAD Data. (First pass on data)
            print(
                f'[1 / 14] Lazy Loading and Preprocessing training data: read_only_few: {self.read_only_few} , sample_triples_ratio: {self.sample_triples_ratio}...',
            )
            self.train_set = load_data_parallel(self.data_dir + '/train.txt', self.read_only_few,
                                                self.sample_triples_ratio,
                                                backend=self.backend)
            print('Train Dataset:', self.train_set)
            print('Done !\n')
            print(
                f'[2 / 14] Lazy Loading and Preprocessing valid data...',
            )
            self.valid_set = load_data_parallel(self.data_dir + '/valid.txt', backend=self.backend)
            print('Validation Dataset:', self.valid_set)
            print('Done !\n')
            print(
                f'[3 / 14] Lazy Loading and Preprocessing test data...',
            )
            self.test_set = load_data_parallel(self.data_dir + '/test.txt', backend=self.backend)
            print('Test Dataset:', self.test_set)
            print('Done !\n')
            self.input_is_parquet = False
        """
        return self.train_set, self.valid_set, self.test_set

    def apply_reciprical_or_noise(self) -> None:
        """ (1) Add reciprocal triples (2) Add noisy triples """
        # (1) Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
        if self.add_reciprical and self.eval_model:
            print(
                '[3.1 / 14] Add reciprocal triples to train, validation, and test sets, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}',
            )
            self.train_set = create_recipriocal_triples(self.train_set)
            if self.valid_set is not None:
                self.valid_set = create_recipriocal_triples(self.valid_set)
            if self.test_set is not None:
                self.test_set = create_recipriocal_triples(self.test_set)
            print('Done !\n')

        # (2) Extend KG with triples where entities and relations are randomly sampled.
        if self.add_noise_rate is not None:
            print(f'[4 / 14] Adding noisy triples...')
            self.train_set = add_noisy_triples(self.train_set, self.add_noise_rate)
            print('Done!\n')

    def deserialize(self, storage_path: str) -> None:
        """ Deserialize data """
        print(f'Deserialization Path Path: {storage_path}\n')
        start_time = time.time()
        print('[1 / 4] Deserializing compressed entity integer mapping...')
        self.entity_to_idx = pd.read_parquet(storage_path + '/entity_to_idx.gzip')  # .compute()
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        self.num_entities = len(self.entity_to_idx)

        print('[2 / ] Deserializing compressed relation integer mapping...')
        start_time = time.time()
        self.relation_to_idx = pd.read_parquet(storage_path + '/relation_to_idx.gzip')
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')

        self.num_relations = len(self.relation_to_idx)
        print(
            '[3 / 4] Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...',
        )
        start_time = time.time()
        self.entity_to_idx = self.entity_to_idx.to_dict()['entity']
        self.relation_to_idx = self.relation_to_idx.to_dict()['relation']
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        # 10. Serialize (9).
        print('[4 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
        start_time = time.time()
        self.train_set = ddf.read_parquet(storage_path + '/idx_train_df.gzip').values.compute()
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
        try:
            print('[5 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
            self.valid_set = ddf.read_parquet(storage_path + '/idx_valid_df.gzip').values.compute()
            print('Done!\n')
        except FileNotFoundError:
            print('No valid data found!\n')
            self.valid_set = None  # pd.DataFrame()

        try:
            print('[6 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
            self.test_set = ddf.read_parquet(storage_path + '/idx_test_df.gzip').values.compute()
            print('Done!\n')
        except FileNotFoundError:
            print('No test data found\n')
            self.test_set = None

    @property
    def entities_str(self) -> List:
        return list(self.entity_to_idx.keys())

    @property
    def relations_str(self) -> List:
        return list(self.relation_to_idx.keys())
