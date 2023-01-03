from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestPolyak:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'pandas'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_rdf_as_backend(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/Family'
        args.backend = 'pandas'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'modin'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_modin_rdf_as_backend(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/Family'
        args.backend = 'modin'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'polars'
        Execute(args).start()
