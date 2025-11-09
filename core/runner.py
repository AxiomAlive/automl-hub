import itertools
import logging
import pprint
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Union, Optional, List, final

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from core.automl import Imbaml, AutoGluon, FLAML
from core.domain import TabularDataset, MLTask
from core.preprocessing import TabularDatasetPreprocessor
from benchmark.repository import FittedModel, ZenodoRepository, TabularDatasetRepository
from utils.decorators import Decorators

logger = logging.getLogger(__name__)


class AutoMLRunner(ABC):
    def __init__(
        self,
        automl,
        metric,
        log_to_file,
        *args,
        **kwargs
    ):
        def _validate_metrics(metric: str):
            if metric not in ['f1', 'bal_acc', 'ap']:
                raise ValueError(
                    f"""
                    Invalid value of metric parameter: {metric}.
                    Available options: ['f1', 'bal_acc', 'ap'].
                    """)
        if isinstance(metric, list):
            for m in metric:
                _validate_metrics(m)
            self._metrics = metric
        else:
            _validate_metrics(metric)
            self._metrics = [metric]

        if automl == 'ag':
            self._automl = AutoGluon(*args, **kwargs)
        elif automl == 'flaml':
            self._automl = FLAML()
        elif automl == 'imbaml':
            self._automl = Imbaml(*args, **kwargs)
        else:
            raise ValueError(
                f"""
                Invalid value of automl parameter: {automl}.
                Options available: ['ag', 'flaml', 'imbaml'].
                """)
        
        self._fitted_model: Optional[FittedModel]
        self._log_to_file = log_to_file

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()

    def _configure_environment(self) -> None:
        logging_handlers = [
            logging.StreamHandler(stream=sys.stdout),
        ]

        if self._log_to_file:
            log_filepath = 'logs/'
            Path(log_filepath).mkdir(parents=True, exist_ok=True)
            log_filepath += datetime.now().strftime(f'{self._automl} {",".join(self._metrics)} %Y.%m.%d %H:%M:%S')
            log_filepath += '.log'
            logging_handlers.append(logging.FileHandler(filename=log_filepath, encoding='utf-8', mode='w'))

        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(message)s',
            handlers=logging_handlers
        )

        logger.info(f"Optimization metrics are {self._metrics}.")

    @final
    def _run_on_dataset(self, dataset: TabularDataset) -> None:
        if dataset is None:
            logger.error("Run failed. Reason: dataset is undefined.")
            return

        if isinstance(dataset.X, np.ndarray) or isinstance(dataset.X, pd.DataFrame):
            preprocessor = TabularDatasetPreprocessor()
            preprocessed_data = preprocessor.preprocess_data(dataset.X, dataset.y.squeeze())

            assert preprocessed_data is not None

            X, y = preprocessed_data
            X_train, X_test, y_train, y_test = preprocessor.split_data_on_train_and_test(X, y.squeeze())
        else:
            raise TypeError(f"pd.DataFrame or np.ndarray was expected. Got: {type(dataset.X)}")

        logger.info(f"{dataset.id}...Loaded dataset name: {dataset.name}.")
        logger.debug(f'Rows: {X_train.shape[0]}. Columns: {X_train.shape[1]}')

        
        class_belongings = Counter(y_train)
        logger.info(class_belongings)

        if len(class_belongings) > 2:
            raise ValueError("Multiclass problems currently not supported.")

        iterator_of_class_belongings = iter(sorted(class_belongings))
        *_, positive_class_label = iterator_of_class_belongings
        logger.debug(f"Inferred positive class label: {positive_class_label}.")

        number_of_positives = class_belongings.get(positive_class_label)
        if number_of_positives is None:
            raise ValueError("Unknown positive class label.")

        training_dataset = TabularDataset(
            id=dataset.id,
            name=dataset.name,
            X=X_train,
            y=y_train,
            y_label=dataset.y_label
        )

        training_dataset_size = int(pd.DataFrame(X_train).memory_usage(deep=True).sum() / (1024 ** 2))
        training_dataset.size = training_dataset_size
        logger.debug(f"Train sample size is approximately {training_dataset.size} mb.")

        id = itertools.count(start=1)
        for metric in self._metrics:
            task  = MLTask(
                id=next(id),
                dataset=training_dataset,
                metric=metric
            )

            start_time = time.time()
            self._automl.fit(task)
            time_passed = time.time() - start_time
            
            logger.info(f"Training successfully finished.")
            logger.info(f"Training time is {time_passed // 60} min.")

            y_predicted = self._automl.predict(X_test)
            self._automl.score(metric, y_test, y_predicted, positive_class_label)

# TODO: support presets and leaderboard.
class AutoMLSingleRunner(AutoMLRunner):
    def __init__(
        self,
        dataset: TabularDataset,
        automl = 'ag',
        metric: Union[str, List[str]] = 'f1',
        log_to_file = False,
        *args,
        **kwargs
    ):
        super().__init__(automl, metric, log_to_file, *args, **kwargs)
        self._dataset = dataset

        self._configure_environment()

    @Decorators.log_exception
    def run(self) -> None:
        self._run_on_dataset(self._dataset)


class AutoMLBenchmarkRunner(AutoMLRunner):
    def __init__(
        self,
        automl = 'ag',
        metric: Union[str, List[str]] = 'f1',
        log_to_file = True,
        repository: TabularDatasetRepository = ZenodoRepository(),
        *args,
        **kwargs
    ):
        super().__init__(automl, metric, log_to_file, *args, **kwargs)
        self._repository = repository

        self._configure_environment()

    @property
    def repository(self):
        return self._repository
    
    @repository.setter
    def repository(self, value):
        self._repository = value

    @Decorators.log_exception
    def run(self) -> None:
        for dataset in self._repository.datasets:
            self._run_on_dataset(dataset)

