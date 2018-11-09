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

    '''

    def __init__(self,
                 n_output,
                 layers,
                 ):

        ''' Specify the general neural network architecture

        Arguments:
            n_output (`int`)
                Output dimension of the neural network. This should be equal
                to the number of properties to be predicted simultaneously.

            layers (`list` of `int`)
                A list of integers with length equal to the number of hidden
                layers required. Integer values are the number of neurons
                within a given layer.
        '''

        self.n_output = n_output
        self.layers = layers


    def load_data(self,
                  filepath,
                  smiles_col,
                  y_cols,
                  sep=' '):
        '''Load training data from .csv files

        Arguments:
            filepath (`str`):
                Path to .csv file containing training data

            smiles_col (`str`):
                Column name containing SMILES strings

            y_cols (`list` of `str`):
                List of column names containing property data to be learned

            sep (`str`, optional):
                Separator used in .csv file. Defaults to ' '.
        '''

        data = pd.read_csv(filepath, sep=sep)#.head(2000) ####### remove head eventually
        self.smiles = data[smiles_col]
        self.y = np.column_stack((data[y].values for y in y_cols))


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


    def build_network(self, activation='relu', dropout=None):
        '''Prepare neural network graph

        Arguments:
            activation (`str`, optional):
                Activation function to use for all layers. Defaults to 'relu'.

            dropout (`float`, optional):
                Dropout fraction to be applied to all hidden layers. Defaults
                to `None`, where no dropout is applied.
        '''

        if activation == 'relu':
            act = tf.nn.relu
        if activation == 'linear':
            act = tf.nn.linear

        network = []
        for i in range(0, len(self.layers)):
                network.append(tf.keras.layers.Dense(self.layers[i], activation=act))
                if dropout is not None:
                    network.append(tf.keras.layers.Dropout(dropout))
        network.append(tf.keras.layers.Dense(self.n_output, activation=act))
        self.model = tf.keras.Sequential(network)


    def train(self,
              learning_rate=0.01,
              epochs=10,
              batch_size=50,
              loss='mean_absolute_error',
              optimizer='sgd'):

        '''Train neural network

         Arguments:
             learning_rate (`float`, optional):
                Learning rate. Defaults to 0.01.

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

        '''
        tf.keras.backend.clear_session()
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(self.x_train, self.y_train, epochs=epochs)
        #self.model.summary()


    def evaluate(self):
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
        print(string)

        return self.y_test, pred
