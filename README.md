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
nn.load_data('example_data.pkl', 'SMILES', ['PROP1, 'PROP2', 'PROP3'], from_pkl=True)
nn.fingerprint(bits=2048, rad=2, test_frac=0.3)
nn.build_network(2, 256, dropout=0.5, activation='relu', input_dropout=0.5)
nn.train(epochs=10, batch_size=50, loss='mean_absolute_error)
y, pred, mae, rmse = nn.evaluate()
```
Where `'SMILES'` is the name of the data column containing SMILES data and `['PROP1', 'PROP2', 'PROP3']` is a list of columns which contain the training target data (in this case, the model will be trained to predict three different properties). 

### Hyperpearameter optimization using a random search
```python
from pychemlp import MLP

nn = MLP()
nn.load_data('./example_data.pkl', 'SMILES', ['PROP1', 'PROP2', 'PROP3'], from_pkl=True)
nn.fingerprint(bits=2048, rad=2, test_frac=0.3)

search_space = {'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'input_dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'n_layers': [1, 2, 3, 4],
                'n_neurons': [32, 64, 128, 256, 512],
                'learning_rate': [0.1, 0.01, 0.001],
                'batch_size': [32, 64, 128, 256]}
nn.hyperparam_opt_random(search_space, 20, epochs=20)
```
### Using a trained model
Once trained a model can be called as an attribute of the MLP class to make new predictions:
```python
predictions = nn.model.predict(fingerprint_array)
```
Where fingerprint_array is an array of (Morgan) fingerprints which can be obtained by calling `load_data` and `fingerprint`, using a file containing SMILES for which predictions are to be made.

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
