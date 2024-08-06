import numpy as np
import os
from utils.qm9_data_load import DataContainer

dataset = "qm9_eV.npz"
cutoff = 5.0
targets = ['U0']  # ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

data_container = DataContainer(dataset, cutoff=cutoff, target_keys=targets)

b = data_container.__getitem__(0)