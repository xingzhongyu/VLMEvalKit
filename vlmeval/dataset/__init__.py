import os.path as osp
import warnings

from vlmeval.smp import LMUDataRoot, load

from .image_base import ImageBaseDataset, img_root_map
from .manufacture_mcq import (
    ManufactureMCQDatasetQ1,
    ManufactureMCQDatasetQ2,
    ManufactureMCQDatasetQ3,
    ManufactureMCQDatasetQ4,
    ManufactureMCQDatasetQ5,
    combine_manufacture_q_results,
)
from .utils import DEBUG_MESSAGE, build_judge
from .video_dataset_config import supported_video_datasets

CUSTOM_DATASET = [
    ManufactureMCQDatasetQ1,
    ManufactureMCQDatasetQ2,
    ManufactureMCQDatasetQ3,
    ManufactureMCQDatasetQ4,
    ManufactureMCQDatasetQ5,
]

DATASET_CLASSES = CUSTOM_DATASET

SUPPORTED_DATASETS = []
for dataset_cls in DATASET_CLASSES:
    SUPPORTED_DATASETS.extend(dataset_cls.supported_datasets())


def DATASET_TYPE(dataset, *, default: str = 'MCQ') -> str:
    if dataset is None:
        return default
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            return getattr(cls, 'TYPE', default)
    warnings.warn(f'Dataset {dataset} is not officially supported. Will treat it as {default}.')
    return default


def DATASET_MODALITY(dataset, *, default: str = 'IMAGE') -> str:
    if dataset is None:
        return default
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            return getattr(cls, 'MODALITY', default)
    warnings.warn(f'Dataset {dataset} is not officially supported. Will treat it as {default}.')
    return default


def build_dataset(dataset_name, **kwargs):
    if dataset_name in supported_video_datasets:
        return supported_video_datasets[dataset_name](**kwargs)

    for cls in DATASET_CLASSES:
        if dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)

    warnings.warn(f'Dataset {dataset_name} is not officially supported.')
    data_file = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    if not osp.exists(data_file):
        warnings.warn(f'Data file {data_file} does not exist. Dataset building failed.')
        return None

    data = load(data_file)
    if 'question' not in [str(x).lower() for x in data.columns]:
        warnings.warn(f'Data file {data_file} does not have a `question` column. Dataset building failed.')
        return None

    warnings.warn(
        f'Found a TSV for {dataset_name}, but this benchmark-focused build only supports the manufacturing datasets.'
    )
    return None


def infer_dataset_basename(dataset_name):
    return "_".join(dataset_name.split("_")[:-1])


__all__ = [
    'build_dataset',
    'img_root_map',
    'build_judge',
    'DEBUG_MESSAGE',
    'DATASET_TYPE',
    'DATASET_MODALITY',
    'SUPPORTED_DATASETS',
    'supported_video_datasets',
    'infer_dataset_basename',
    'ImageBaseDataset',
    'ManufactureMCQDatasetQ1',
    'ManufactureMCQDatasetQ2',
    'ManufactureMCQDatasetQ3',
    'ManufactureMCQDatasetQ4',
    'ManufactureMCQDatasetQ5',
    'combine_manufacture_q_results',
]
