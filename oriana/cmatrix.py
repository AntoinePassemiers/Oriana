# -*- coding: utf-8 -*-
# cmatrix.py: Count matrices
# author : Antoine Passemiers

from oriana.exceptions import DatatypeException

import numpy as np
import scipy.sparse
import pandas as pd


class CountMatrix:
    """Count matrix.

    Element (i, j) gives the number of occurences of 
    object j in sample i.

    Attributes:
        __data (:obj:`pd.DataFrame`): Dataframe storing the counts
    """

    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = pd.Dataframe(data=data)
        else:
            raise DatatypeException(
                    'Incompatible type %s' % type(data))

    def as_array(self):
        """Converts count matrix to NumPy array.

        returns:
            :obj:`np.ndarray`: NumPy array containing the counts
        """
        return self._data.values

    def as_sparse_matrix(self, mode='csc'):
        """ Converts count matrix to sparse matrix.

        Parameters:
            mode (str): Type of sparse csc_matrix.
                Either 'csc' or 'csr'.

        returns:
            object: A scipy sparse matrix
        """
        arr = self.as_array()
        if mode == 'csc':
            mat = scipy.sparse.csc_matrix(arr)
        else:
            mat = scipy.sparse.csc_matrix(arr)
        return mat

    @staticmethod
    def from_csv(filepath, delimiter=',', has_col_names=True,
                 has_row_names=True):
        """Reads count matrix from file.

        Parameters:
            filepath (str): Path to a csv file
            delimiter (str): Data delimiter (example: ' ')
            has_col_names (bool): Whether csv file has a column header
            has_row_names (bool): Whether first column of the csv file
                contains row names

        Returns:
            :obj:`oriana.CountMatrix`: Count matrix
        """
        df = pd.read_csv(
                filepath,
                sep=delimiter,
                header=(0 if has_col_names else None),
                index_col=(0 if has_row_names else False),
                skip_blank_lines=True)
        return CountMatrix(df)

    @property
    def T(self):
        return CountMatrix(self._data.transpose(copy=False))
   
    @property
    def col_names(self):
        """Column names.

        Returns:
            :obj:´np.ndarray`: Array of object names
        """
        return self._data.columns.values

    @property
    def row_names(self):
        """Row names.

        Returns:
            :obj:`np.ndarray`: Array of sample names
        """
        return np.asarray(self._data.index)
    
    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return self._data.__repr__()
