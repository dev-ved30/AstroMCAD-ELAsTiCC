import os
import math

import numpy as np
import polars as pl

from LSST_Source import LSST_Source
from taxonomy import get_astrophysical_class

# All samples in the same batch need to have consistent sequence length. This adds padding for sequences shorter than sequence_length and truncates light sequences longer than sequence_length 
sequence_length = 500

# Value used for padding tensors to make them the correct length
padding_constant = 0

class LSSTSourceDataSet():


    def __init__(self, path):
        """
        Arguments:
            path (string): Directory with all the astropy tables.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        print(f'Loading parquet dataset: {path}', flush=True)

        self.path = path
        self.parquet = pl.read_parquet(path)
        self.num_sample = self.parquet.shape[0]

        print(f"Number of sources: {self.num_sample}")

    def __len__(self):

        return self.num_sample

    def __getitem__(self, idx):
        
        row = self.parquet[idx]
        source = LSST_Source(row)

        return source
    
    def get_dimensions(self):

        idx = 0
        ts_np, static_np, class_labels, _ = self.__getitem__(idx)

        dims = {
            'ts': ts_np.shape[1],
            'static': static_np.shape[0],
        }

        return dims
    
    def get_labels(self):

        ELASTICC_labels = self.parquet['ELASTICC_class']
        astrophysical_labels = []

        for idx in range(self.num_sample):

            elasticc_class = ELASTICC_labels[idx]
            astrophysical_class = get_astrophysical_class(elasticc_class)
            astrophysical_labels.append(astrophysical_class)
        
        return astrophysical_labels

