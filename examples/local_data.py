import logging
import pandas as pd
from core.runner import AutoMLSingleRunner
from utils.helpers import make_tabular_dataset

logger = logging.getLogger(__name__)


def main():
    df = pd.read_csv("datasets/local/kc1-binary.csv")
    y = df["DL"]
    X = df.drop(["DL"], axis=1)
    dataset = make_tabular_dataset(
        name="kc1-binary",
        X=X,
        y=y,
        y_label="DL"
    )

    automl = AutoMLSingleRunner(dataset, metric='f1', automl='ag')
    automl.run()

if __name__ == '__main__':
    main()
