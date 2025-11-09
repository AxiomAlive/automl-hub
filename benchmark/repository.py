import itertools
import logging
import multiprocessing
import os
import pprint
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, TypeVar

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets

from core.domain import TabularDataset
from utils.helpers import make_tabular_dataset

logger = logging.getLogger(__name__)
FittedModel = TypeVar('FittedModel', bound=Any)


class TabularDatasetRepository(ABC):
    def __init__(self, *args, **kwargs):
        self._datasets: List[TabularDataset] = []

    @abstractmethod
    def load_datasets(self, id_range: Optional[List[int]] = None) -> List[TabularDataset]:
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, id: Optional[int] = None) -> TabularDataset:
        raise NotImplementedError()

    @property
    def datasets(self):
        return self._datasets


class ZenodoRepository(TabularDatasetRepository):
    def __init__(self):
        super().__init__()
        self._raw_datasets = fetch_datasets(data_home='datasets/imbalanced-learning', verbose=True)

    def load_dataset(self, id: Optional[int] = None) -> TabularDataset:
        for i, (dataset_name, dataset_data) in enumerate(self._raw_datasets.items(), 1):
            if i == id:
                return make_tabular_dataset(
                    name=dataset_name,
                    X=dataset_data.get("data"),
                    y=dataset_data.get("target")
                )
            elif i > id:
                raise ValueError(f"TabularDataset(id={id}) is not available.")

    def load_datasets(self, id_range: Optional[List[int]] = None) -> List[TabularDataset]:
        if id_range is None:
            range_start = 1
            range_end = len(self._raw_datasets.keys()) + 1
            id_range = range(range_start, range_end)
            logger.info(f"Running tasks from {range_start} to {range_end}.")
        for i in id_range:
            self._datasets.append(self.load_dataset(i))
        return self.datasets


# TODO: refactor.
class OpenMLRepository(TabularDatasetRepository):
    def __init__(self, suite_id=271):
        super().__init__()
        self._suite_id = suite_id
        import openml
        openml.config.set_root_cache_directory("openml_cache")

    def load_dataset(self, id: Optional[int] = None) -> TabularDataset:
        try:
            with multiprocessing.Pool(processes=1) as pool:
                task = pool.apply_async(openml.tasks.get_task, [id]).get(timeout=1800)
                dataset = pool.apply_async(task.get_dataset, []).get(timeout=1800)
            X, y, categorical_indicator, dataset_feature_names = dataset.get_data(
                target=dataset.default_target_attribute)

        except multiprocessing.TimeoutError:
            logger.error(f"Fetch from OpenML timed out. TabularDataset id={id} was not loaded.")
            raise multiprocessing.TimeoutError()
        except Exception as exc:
            logger.error(pprint.pformat(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            raise exc

        return make_tabular_dataset(
            name=dataset.name,
            y_label=dataset.default_target_attribute,
            X=X,
            y=y
        )

    def load_datasets(self, id_range: List[int] = None) -> List[TabularDataset]:
        benchmark_suite = openml.study.get_suite(suite_id=self._suite_id)
        for i, id in enumerate(benchmark_suite.tasks):
            if id_range is not None and i not in id_range:
                continue
            self._datasets.append(self.load_dataset(id))
        return self.datasets
