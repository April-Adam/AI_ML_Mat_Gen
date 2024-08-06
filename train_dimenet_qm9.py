import tensorflow as tf
import numpy as np
import os
import ast
import logging
import string
import random
import yaml
from datetime import datetime

from dimenet.model.dimenet import DimeNet
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish

from utils.trainer import Trainer
from utils.metrics import Metrics
from utils.qm9_data_load import DataContainer
from utils.qm9_data_provider import DataProvider



dataset    = "qm9_eV.npz"
targets    = ['U0']
cutoff     = 5.0
num_train  = 110000
num_valid  = 10000
data_seed  = 42
batch_size = 32

train = {}
validation = {}

train['metrics'] = Metrics('train', targets)
validation['metrics'] = Metrics('val', targets)

data_container = DataContainer(dataset, cutoff=cutoff, target_keys=targets)

# Initialize DataProvider (splits dataset into 3 sets based on data_seed and provides tf.datasets)
data_provider = DataProvider(data_container, num_train, num_valid, batch_size,
                             seed=data_seed, randomized=True)

# Initialize datasets
train['dataset'] = data_provider.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE)
train['dataset_iter'] = iter(train['dataset'])
validation['dataset'] = data_provider.get_dataset('val').prefetch(tf.data.experimental.AUTOTUNE)
validation['dataset_iter'] = iter(validation['dataset'])


emb_size          = 128
num_blocks        = 6
num_bilinear      = 8
num_spherical     = 7
num_radial        = 6
envelope_exponent = 5
num_before_skip   = 1
num_after_skip    = 2
num_dense_output  = 3
output_init       = 'GlorotOrthogonal'

model_name = "dimenet"

if model_name == "dimenet":
    model = DimeNet(
            emb_size=emb_size, num_blocks=num_blocks, num_bilinear=num_bilinear,
            num_spherical=num_spherical, num_radial=num_radial,
            cutoff=cutoff, envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip, num_after_skip=num_after_skip,
            num_dense_output=num_dense_output, num_targets=len(targets),
            activation=swish, output_init=output_init)
elif model_name == "dimenet++":

    out_emb_size   = 256
    int_emb_size   = 64
    basis_emb_size = 8
    extensive      = True

    model = DimeNetPP(
            emb_size=emb_size, out_emb_size=out_emb_size,
            int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
            num_blocks=num_blocks, num_spherical=num_spherical, num_radial=num_radial,
            cutoff=cutoff, envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip, num_after_skip=num_after_skip,
            num_dense_output=num_dense_output, num_targets=len(targets),
            activation=swish, extensive=extensive, output_init=output_init)
else:
    raise ValueError(f"Unknown model name: '{model_name}'")

num_steps = 3000000
ema_decay = 0.999

learning_rate = 0.001
decay_rate = 0.01
decay_steps = 4000000

trainer = Trainer(model, learning_rate, 
                  decay_steps, decay_rate,
                  ema_decay=ema_decay, max_grad_norm=1000)

step_init = 1
evaluation_interval = 10
steps_per_epoch = int(np.ceil(num_train / batch_size))

for step in range(step_init, num_steps + 1):
    print("training: ", step)
    # Perform training step
    trainer.train_on_batch(train['dataset_iter'], train['metrics'])


    if (step % evaluation_interval == 0):

            # Compute results on the validation set
            for i in range(int(np.ceil(num_valid / batch_size))):
                trainer.test_on_batch(validation['dataset_iter'], validation['metrics'])

            epoch = step // steps_per_epoch
            print(
                f"{step}/{num_steps} (epoch {epoch+1}): "
                f"Loss: train={train['metrics'].loss:.6f}, val={validation['metrics'].loss:.6f}; "
                f"logMAE: train={train['metrics'].mean_log_mae:.6f}, "
                f"val={validation['metrics'].mean_log_mae:.6f}")


