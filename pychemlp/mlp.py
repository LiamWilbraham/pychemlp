import random
import tensorflow as tf
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

class MLP:
    '''Class to construct, train and evaluate Multylayer Perceptrons.

    Simple neural networks can be constructed, trained and evaluated to
    predict molecular properties directly from SMILES strings.

    Methods:
        load_data:
            Import training data from .csv files

        fingerprint:
            Generate machine-learnable molecular fingerprints from SMILES.
            Optionally split data into training and testing groups.

        build_network:
            Construct neural network

        train:
            Train neural network to predict an arbitrary number of properties

        evaluate:
            Evaluate neural network performance on test data group.

        hyperparam_opt_random:
            Optimize hyperparameters using a random search strategy.

    '''

    def __init__(self, name='mlp'):
        ''' Specify the general neural network architecture

        Arguments:
            name (`str`, optional)
                Give a name to the MLP class instance.
        '''
        self.name = name


    def load_data(self,
                  filepath,
                  smiles_col,
                  y_cols,
                  sep=' ',
                  from_pkl=False):
        '''Load training data from .csv files

        Arguments:
            filepath (`str`):
                Path to file containing training data

            smiles_col (`str`):
                Column name containing SMILES strings

            y_cols (`list` of `str`):
                List of column names containing property data to be learned

            sep (`str`, optional):
                Separator used in .csv file. Defaults to ' '.

            from_pkl (`bool`, optional):
                Load data from pickled object. Defaults to False.
        '''

        if from_pkl:
            self.data = pd.read_pickle(filepath)

        else:
            self.data = pd.read_csv(filepath, sep=sep)

        self.smiles = self.data[smiles_col]
        self.y = np.column_stack((self.data[y].values for y in y_cols))
        self.n_output = len(y_cols)


    def fingerprint(self, bits=512, rad=2, test_frac=None):
        '''Fingerprint molecules from SMILES strings

        Arguments:
            bits (`int`, optional):
                Length of bit string used to represent molecular fingerprint.
                Note that this will therefore be the dimension of the input
                to the neural network.

            rad (`int`, optional):
                Maximum radius of molecular fragments to consider when
                fingerprinting.

            test_frac (`float`, optional):
                Fraction of molecules to be used for model evaluation.
                Defaults to `None`, indicating no test fraction.
        '''
        def morgan(smi, rad, bits):
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, rad, nBits=bits)
            return np.array(fp)

        x = np.zeros((len(self.smiles), bits))
        for index, smi in enumerate(self.smiles):
            x[index] = morgan(smi, rad, bits)

        if test_frac is not None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, self.y, test_size=test_frac)
        else:
            self.x_train = x
            self.y_train= self.y


    def build_network(self,
                      n_layers,
                      n_neurons,
                      activation='relu',
                      dropout=None,
                      input_dropout=None):
        '''Prepare neural network graph

        Arguments:
            n_layers (`int`):
                Number of hidden layers.

            n_neurons (`int`):
                Number of neurons per hidden layer.

            activation (`str`, optional):
                Activation function to use for all layers. Defaults to 'relu'.

            dropout (`float`, optional):
                Dropout fraction to be applied to all hidden layers. Defaults
                to `None`, where no dropout is applied.

            input_dropout (`float`, optional):
                Dropout fraction to be applied to the input layer. Defaults
                to `None`, where no dropout is applied.
        '''
        tf.keras.backend.clear_session()

        if activation == 'relu':
            act = tf.nn.relu
        if activation == 'linear':
            act = tf.nn.linear

        network = []
        if input_dropout is not None:
            network.append(tf.keras.layers.Dropout(input_dropout))
        for i in range(0, n_layers):
                network.append(tf.keras.layers.Dense(n_neurons, activation=act))
                if dropout is not None:
                    network.append(tf.keras.layers.Dropout(dropout))
        network.append(tf.keras.layers.Dense(self.n_output, activation=act))
        self.model = tf.keras.Sequential(network)


    def train(self,
              epochs=10,
              batch_size=50,
              loss='mean_absolute_error',
              optimizer='adam',
              learning_rate=0.001,
              decay=0.0,
              validation_split=0.0):
        '''Train neural network

         Arguments:
             epochs (`int`, optional):
                Number of training epochs. Defaults to 10.

             batch_size (`int`, optional):
                Batch size to be used in training. Defaults to 50.

             loss (`str`, optional):
                Loss function to be used in training. Defaults to
                'mean_absolute_error'.

             optimizer (`str`, optional):
                Optimizer to be used in training. Defaults to 'sgd'
                (stochastic gradient descent).

             learning_rate (`float`, optional):
                Learning rate to be used by the optimizer. Defaults to
                `0.001`.

             decay (`float`, optional):
                Learning rate decay over each update used by the optimizer.
                Defaults to '0.0'
        '''
        if optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=learning_rate, decay=decay)
        elif optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay)

        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(self.x_train, self.y_train, epochs=epochs,
                        validation_split=validation_split, verbose=0)
        #self.model.summary()


    def evaluate(self, silent=False):
        '''Evaluate model performance against test data

        Returns:
            y (`array`):
                Target data from test group.

            pred (`array`):
                Model's prediction of target data.

        '''

        pred = self.model.predict(self.x_test)
        mae = np.mean(np.abs(self.y_test - pred))
        rmse = np.sqrt(np.mean(((self.y_test - pred)**2)))

        string = '\nModel Evaluation\n'
        string += '----------------\n'
        string += 'Mean Absolute Error: {:.4}\n'.format(mae)
        string += 'RMS Error : {:.4}\n'.format(rmse)

        if not silent:
            print(string)

        return self.y_test, pred, mae, rmse


    def hyperparam_opt_random(self, search_space, iterations, epochs=10):
        '''Optimize hyperparameters via a random search

        search_space (`dict`):
            Dictionary specifying the search space. Keys are the hyperparameters
            to be optimized and values are lists of allowed hyperparameter
            values. Allowed hyperparameters to be otimised are 'n_layers',
            'n_neurons', 'dropout', 'input_dropout', 'activation', 'batch_size',
            'optimizer', 'loss', 'learning_rate' & 'decay'.

        iterations (`int`):
            Number of iterations of random search to be performed.

        epochs (`int`, optional):
            Number of epochs to be used in training during each iteration
            of the random search.
        '''

        p = {'n_layers': 2,
             'n_neurons': 256,
             'dropout': 0.0,
             'input_dropout': 0.0,
             'activation': 'relu',
             'batch_size': 32,
             'optimizer': 'adam',
             'loss': 'mean_absolute_error',
             'learning_rate': 0.01,
             'decay': 0.00,}

        mse_best = 10**10
        for i in range(iterations):
            for key, value in search_space.items():
                random_value = random.choice(value)
                p[key] = random_value

            self.build_network(n_layers=p['n_layers'],
                               n_neurons=p['n_neurons'],
                               dropout=p['dropout'],
                               input_dropout=p['input_dropout'],
                               activation=p['activation'])

            self.train(epochs=epochs,
                       batch_size=p['batch_size'],
                       optimizer=p['optimizer'],
                       loss=p['loss'],
                       learning_rate=p['learning_rate'],
                       decay=p['decay'],
                       )

            y, pred, mae, mse = self.evaluate(silent=True)

            print('Iteration : {:03}, RMSE: {:.4f}'.format(i, mse))

            if mse < mse_best:
                mse_best = mse
                p_best = p
                self.best_model = self.model

        string = '\nBest Model, RMSE = {:.4f}\n'.format(mse_best)
        string += '-----------------------\n'
        for key, value in p_best.items():
            string += ('{} : {}\n'.format(key, value))
        print(string)
