from pychemlp import MLP

nn = MLP()
nn.load_data(example_data.pkl, 'TRIMER', ['IP (eV)', 'EA (eV)', 'Excitation Energy (eV)'], from_pkl=True)
nn.fingerprint(bits=2048, rad=2, test_frac=0.3)
nn.build_network(2, 256, dropout=0.5)
nn.train()
y, pred, mae, rmse = nn.evaluate()
