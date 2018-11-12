# pychemlp
Easy-to-use python module for the training of Multi-Layer Perceptrons (neural networks) from molecular SMILES data and known associated properties.

## Functionality

* import data from external .csv files and pickled objects
* generate molecular fingerprints based on the Extended Connectivity Fingerprint (ECFP) method
* Specify neural network architectures (number of layers, neurons per layer, dropout)
* Train and evaluate models
* Conduct hyperparameter optimization using a random search

## Using the module

### Simple construction and training of a neural network
```python
from pychemlp import MLP
nn = MLP()
nn.load_data('example_data.pkl', 'TRIMER', ['IP (eV)', 'EA (eV)', 'Excitation Energy (eV)'], from_pkl=True)
nn.fingerprint(bits=2048, rad=2, test_frac=0.3)
nn.build_network(2, 256, dropout=0.5)
nn.train()
y, pred, mae, rmse = nn.evaluate()
```
### Hyperpearameter optimization using a random search
```python
search_space = {'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'input_dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'n_layers': [1, 2, 3, 4],
                'n_neurons': [32, 64, 128, 256, 512],
                'learning_rate': [0.1, 0.01, 0.001],
                'batch_size': [32, 64, 128, 256]}

nn = MLP()
nn.load_data('./example_data.pkl', 'TRIMER', ['IP (eV)', 'EA (eV)', 'Excitation Energy (eV)'], from_pkl=True)
nn.fingerprint(bits=2048, rad=2, test_frac=0.3)
nn.hyperparam_opt_random(search_space, 20, epochs=20)
```

## Installation & Requirements

To install, simply clone the repo:
```
git clone https://github.com/LiamWilbraham/pychemlp.git
```

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
