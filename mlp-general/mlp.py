import tensorflow as tf
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self,
            	 n_input,
                 n_output,
                 layers,
                 ):

        self.n_input = n_input
        self.n_output = n_output
        self.layers = layers


    def load_data(self,
                  filepath,
                  smiles_col,
                  y_cols,
                  sep=' '):

        data = pd.read_csv(filepath, sep=sep).head(2000) ####### remove head eventually
        self.smiles = data[smiles_col]
        self.y = np.column_stack((data[y].values for y in y_cols))


    def fingerprint(self, test_frac=None):

        def morgan(smi):
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.n_input)
            return np.array(fp)

        x = np.zeros((len(self.smiles), self.n_input))
        for index, smi in enumerate(self.smiles):
            x[index] = morgan(smi)

        if test_frac is not None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, self.y, test_size=test_frac)
        else:
            self.x_train = x
            self.y_train= self.y


    def build_network(self, activation=tf.nn.relu):

        network = []
        for i in range(0, len(self.layers)):
                network.append(tf.keras.layers.Dense(self.layers[i], activation=activation))
        network.append(tf.keras.layers.Dense(self.n_output, activation=activation))
        self.model = tf.keras.Sequential(network)


    def train(self,
              learning_rate=0.1,
              epochs=10,
              batch_size=1,
              loss='mean_absolute_error',
              optimizer='sgd'):

        self.model.compile(optimizer=optimizer,loss=loss)
        self.model.fit(self.x_train, self.y_train, epochs=epochs)
        self.model.summary()

    def evaluate():
        pass
