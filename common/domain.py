import uuid
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class TabularDataset:
    id: int
    name: str
    X: Union[pd.DataFrame, np.ndarray]
    y: Union[pd.Series, np.ndarray]
    y_label: Optional[str] = None
    size: Optional[int] = None

@dataclass
class MLTask:
    id: int
    dataset: TabularDataset
    metric: str
