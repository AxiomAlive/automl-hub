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

from common.automl import Imbaml, AutoGluon, FLAML
from common.domain import TabularDataset
from common.preprocessing import TabularDatasetPreprocessor
from benchmark.repository import FittedModel, ZenodoRepository, TabularDatasetRepository
from utils.decorators import Decorators

logger = logging.getLogger(__name__)


class AutoMLRunner(ABC):
    def __init__(self, automl='imbaml', log_to_file=True, *args, **kwargs):
        self._fitted_model: Optional[FittedModel]
        self._log_to_file = log_to_file
        if automl == 'imbaml':
            self._automl = Imbaml(*args, **kwargs)
        elif automl == 'ag':
            self._automl = AutoGluon(*args, **kwargs)
        elif automl == 'flaml':
            self._automl = FLAML()
        else:
            raise ValueError(
                """
                Invalid --automl option.
                Options available: ['imbaml', 'ag', 'flaml'].
                """)

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
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=logging_handlers
        )

        logger.info(f"Optimization metrics are {self._metrics}.")

    @final
    def _run_on_dataset(self, dataset: TabularDataset) -> None:
        if dataset is None:
            logger.error("dataset run failed. Reason: dataset is undefined.")
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
        logger.info(f'Rows: {X_train.shape[0]}. Columns: {X_train.shape[1]}')

        
        class_belongings = Counter(y_train)
        logger.info(class_belongings)

        if len(class_belongings) > 2:
            raise ValueError("Multiclass problems currently not supported.")

        iterator_of_class_belongings = iter(sorted(class_belongings))
        *_, positive_class_label = iterator_of_class_belongings
        logger.info(f"Inferred positive class label: {positive_class_label}.")

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
        logger.info(f"Train sample size is {training_dataset_size} mb.")

        training_dataset.size = training_dataset_size

        for metric in self._metrics:
            start_time = time.time()
            self._automl.fit(training_dataset, metric)
            logger.info(f"Training on dataset (id={dataset.id}, name={dataset.name}) successfully finished.")

            time_passed = time.time() - start_time
            logger.info(f"Training time is {time_passed // 60} min.")

            y_predictions = self._automl.predict(X_test)
            # TODO: evaluate on additional metrics for a single runner.
            self._automl.score(metric, y_test, y_predictions, positive_class_label)


class AutoMLSingleRunner(AutoMLRunner):
    def __init__(self, dataset: TabularDataset, metric: str = 'f1', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = [metric]
        self._dataset = dataset

        self._configure_environment()

    @Decorators.log_exception
    def run(self) -> None:
        self._run_on_dataset(self._dataset)


class AutoMLBenchmarkRunner(AutoMLRunner):
    def __init__(self, metrics: Optional[List[str]], repository: TabularDatasetRepository = ZenodoRepository(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        if metrics is None:
            self._metrics = ['f1']
        else:
            self._metrics = metrics
        self._repository = repository

        self._configure_environment()

    @property
    def repository(self):
        return self._repository

    @Decorators.log_exception
    def run(self) -> None:
        for dataset in self._repository.datasets:
            self._run_on_dataset(dataset)

