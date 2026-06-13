# coding: utf-8

from data.datasets_factory.dataset_x2v_interleave import X2VInterleaveIterableDataset
from data.datasets_factory.dataset_x2v_interleave_local import X2VInterleaveLocalIterableDataset
from data.datasets_factory.dataset_x2t_interleave import X2TInterleaveIterableDataset
from data.datasets_factory.dataset_x2t_interleave_local import X2TInterleaveLocalIterableDataset

DATASET_REGISTRY = {
    # GEN
    'x2v_interleave': X2VInterleaveIterableDataset,
    'x2v_interleave_local': X2VInterleaveLocalIterableDataset,
    'x2t_interleave': X2TInterleaveIterableDataset,
    'x2t_interleave_local': X2TInterleaveLocalIterableDataset,
    }
