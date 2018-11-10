# mlp-general
Easy-to-use python module for the training of Multi-Layer Perceptrons (neural networks) from molecular SMILES data and known associated properties.

## Functionality
Within the module, the user can:

* import data from external .csv files and pickled objects
* generate molecular fingerprints based on the Extended Connectivity Fingerprint (ECFP) method
* Specify neural network architectures (number of layers, neurons per layer, dropout)
* Train and evaluate models

## Using the module
```python
from mlp import MLP
nn = MLP(3, [100, 100])
nn.load_data('/home/examples/input-data.csv','smiles', ['property1', 'property2'])
nn.fingerprint(bits=1024, rad=2, test_frac=0.3)
nn.build_network(dropout=0.2)
nn.train()
y_target, y_predicted = nn.evaluate()
```

## Installation & Requirements

The module requires the following packages (all can be installed via conda):

* tensorflow (https://www.tensorflow.org/install/)
* scikit-learn (https://scikit-learn.org/stable/install.html)
* pandas (https://anaconda.org/anaconda/pandas)
* numpy (https://anaconda.org/anaconda/numpy)
* rdkit (https://www.rdkit.org/docs/Install.html)

rdkit may be installed via conda (recommended):
```
conda install -c rdkit rdkit
```
