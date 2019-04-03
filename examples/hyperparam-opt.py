from pychemlp import MLP

search_space = {'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'input_dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'n_layers': [1, 2, 3, 4],
                'n_neurons': [32, 64, 128, 256, 512],
                'learning_rate': [0.1, 0.01, 0.001],
                'batch_size': [32, 64, 128, 256]}

nn = MLP()
<<<<<<< HEAD:examples/train_network.py
nn.load_data('example_data.pkl', 'TRIMER', ['IP (eV)', 'EA (eV)', 'Excitation Energy (eV)'], from_pkl=True)
nn.fingerprint(bits=2048, rad=2, test_frac=0.3)
nn.hyperparam_opt_random(search_space, 20, epochs=20)
=======
nn.load_data('training-data.pkl', 'TRIMER', ['Calib. IP (eV)', 'Calib. EA (eV)', 'Calib. Excitation Energy (eV)'])
nn.fingerprint(bits=2048, rad=2, test_frac=0.5)
nn.hyperparam_opt_random(search_space, 100, epochs=20)
>>>>>>> def9f65ee5694f9f5b6393e8c5aa2be7ee634c7b:examples/hyperparam-opt.py
