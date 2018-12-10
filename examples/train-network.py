from pychemlp import MLP

nn = MLP()
nn.load_data('training-data.csv', 'TRIMER', ['Calib. IP (eV)', 'Calib. EA (eV)', 'Calib. Excitation Energy (eV)'])
nn.fingerprint(bits=2048, rad=2, test_frac=0.5)
nn.build_network(2, 256, dropout=0.5)
nn.train()
y, pred, mae, rmse = nn.evaluate()
