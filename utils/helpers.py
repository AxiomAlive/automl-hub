import itertools
from typing import Optional, Union

import numpy as np
import pandas as pd
from core.domain import TabularDataset

id = itertools.count(start=1)


def make_tabular_dataset(
    name: str,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    y_label: Optional[str] = None,
    size: Optional[int] = None
) -> TabularDataset:
    return TabularDataset(
        id=next(id),
        name=name,
        X=X,
        y=y,
        y_label=y_label,
        size=size
    )