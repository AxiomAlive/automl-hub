# AutomlHub
This tool allows to run popular AutoML's in a unified way.

### Project status
Currently, only binary imbalanced classification setting is implemented.

AutoML options include [Imbaml](https://github.com/AxiomAlive/imbaml), [Auto-gluon](https://github.com/autogluon/autogluon) and [FLAML](https://github.com/microsoft/FLAML).

### Usage
Only Linux support has been tested. Support for Windows and MacOS is not confirmed, and you may run into bugs or a suboptimal experience.

To run a [benchmark](https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html#imblearn.datasets.fetch_datasets) locally just type in the terminal:
```
./benchmark.sh
```
By default, benchmark for **Imbaml** will be run.
To change to **AutoGluon** add the `-ag` argument;
to change to **FLAML** add the `-flaml` argument.
<br>
<br>
By default, output is to file and console,
to change it to console only, use the `-c` argument.

[comment]: <> (<br>)

[comment]: <> (<br>)

[comment]: <> (Also, a cloud run option is available &#40;with a `-cloud` argument&#41; on [Yandex.Datasphere]&#40;https://datasphere.yandex.cloud/&#41;. En example of configuration file is `cloud.yaml`.)

#### Example
```python
from benchmark.repository import ZenodoRepository
from common.runner import AutoMLSingleRunner

def main():
    dataset = ZenodoRepository().load_dataset(1)
    automl = AutoMLSingleRunner(dataset, 'f1')
    automl.run()

if __name__ == '__main__':
    main()
```
