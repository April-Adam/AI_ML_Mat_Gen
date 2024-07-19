import tensorflow as tf
import numpy as np
from fetch_data import qm9_parse, qm9_fetch
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
from keras import backend as K
from sklearn.utils import shuffle

graph_feature_len = 8
node_feature_len = 16
msg_feature_len = 16

qm9_records = qm9_fetch()
data = qm9_parse(qm9_records)

def convert_record(d):
    """ Takes a parsed record from the QM9 dataset and extracts the elements, coordinates of the atoms, and the ground truth quantum properties. """

    # break up record
    (elems, coords), labels = d
    elems = elems.numpy() # the atoms in the compound
    coords = coords.numpy()
    r = coords[:, :3] # of the 4 values in each element of elems, the first 3 elements are the xyz coordinates

    # one hot encodes the atoms into a 32x16 np array
    # 32x16: 32 total atoms in the compound, 16 possible elements per atom
    one_hot_encode = np.zeros((32, 16))
    one_hot_encode[np.arange(len(elems)), elems - 1] = 1

    return (one_hot_encode, r), labels.numpy()

def coords_to_distances(x):
    """ Takes the (x, y, z) coordinates of the atoms and outputs the inverse squared distances between pairs of atoms. """

    # Finding squared distances between pairs of atoms:
    # x[:, np.newaxis, :]                                               - creates an array with shape = (N, 1, 3)
    # x - x[:, np.newaxis, :]                                           - finds the difference in the x, y, z coords for each pair of atoms
    # (x - x[:, np.newaxis, :]) ** 2                                    - finds the squares of the above-mentioned differences
    # r2 = np.sum((x - x[:, np.newaxis, :]) ** 2, axis=-1)              - finds the sum of these squares - this gets the squared distance between pairs of atoms
    r2 = np.sum((x - x[:, np.newaxis, :]) ** 2, axis=-1)

    # for each element in `r2`, if nonzero, take its reciprocal (adding sys.float_info.epsilon to prevent divisions by 0)
    e = np.where(r2 != 0.0, 1 / (r2 + sys.float_info.epsilon), 0.0)

    ret = np.zeros((32, 32))
    ret[0:x.shape[0], 0:x.shape[0]] = e # copies the values of `e` to the top left N x N square of `ret`
    return ret

def conv1_block(inputs, feature_dims, activation):
    # Graph Convolution Layer
    outputs = keras.layers.Conv1D(feature_dims, kernel_size=1, activation=None, use_bias=False)(inputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Activation(activation)(outputs)
    return outputs

def conv2_block(inputs, feature_dims, activation):
    # Graph Convolution Layer
    outputs = keras.layers.Conv2D(feature_dims, kernel_size=(1, 1), activation=None, use_bias=False)(inputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Activation(activation)(outputs)
    return outputs

def simple_gnn_model():
    nodes = keras.Input(shape=(32, 32, node_feature_len)) # nodes: shape (32, 32, 16)
    edges = keras.Input(shape=(32, 32)) # edges: shape (32, 32)
    features = keras.Input(shape=(graph_feature_len,))

    # input 1: nodes
    out = conv2_block(nodes, msg_feature_len, 'relu')

    # input 2: edges
    # purpose of the below code is to reshape edges for matrix multiplication later on
    x = keras.ops.expand_dims(edges, axis=-1) # shape (None, 32, 32, 1)
    # duplicate each inverse square distance to form 16 length tensors per atom
    x = keras.ops.repeat(x, msg_feature_len, axis=-1) # shape (None, 32, 32, 16)

    # scale the outputs of the conv block inversely to the distances between atoms
    # this means that the farther 2 atoms are, the less their connection affects each other
    out = keras.layers.Multiply()([out, x])

    # aggregate information about each atom's relationship with all the other atoms
    out = keras.layers.Lambda(lambda x: keras.ops.sum(x, axis=2, keepdims=False), output_shape=(32, 16))(out) # shape (None, 32, 16)

    # extract features about atoms through convolution
    new_nodes = conv1_block(out, node_feature_len, 'relu') # shape (None, 32, 16)

    # aggregate information further
    out = keras.layers.Lambda(lambda x: keras.ops.sum(x, axis=1, keepdims=False), output_shape=(16,))(new_nodes) # shape (None, 16)

    # fully connected layer
    new_features = keras.layers.Dense(graph_feature_len, use_bias=True)(out) # shape (None, 16)

    # below code is for features update -> can add features from previous parts of the model to these features.
    # currently, features is just a bunch of 0s, so technically this does nothing right now,
    # but this can be used in the future if we decide to expand the model.
    # inevitably, we will also need a way to preserve the features from previous parts of the model to use,
    # but that's for later.
    new_features = keras.layers.Lambda(lambda x: x[0] + x[1])([new_features, features])

    # output layer
    # each of the 16 values in the output layer corresponds to one of the quantum properties we're predicting
    pred = keras.layers.Dense(16, use_bias=True)(new_features)

    model = keras.Model(inputs=[nodes, edges, features], outputs=[pred])
    return model

# same as tf.keras.metrics.R2Score but as a function
def r2_keras(y_true, y_pred):
    res = keras.backend.sum(keras.backend.square(y_true - y_pred))
    total = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
    return (1 - res / (total + keras.backend.epsilon())) # add keras.backend.epsilon to ensure no division by 0

class DataGenerator(keras.utils.Sequence):
    def __init__(self, nodes, edges, y_true, batch_size):
        self.nodes, self.edges, self.y_true = np.array(nodes), np.array(edges), np.array(y_true)
        self.batch_size = batch_size
        self.batches = np.ceil(len(nodes) / batch_size)
        self.batch_idx = np.array_split(range(len(nodes)), self.batches) # splits into `num_batches` batches

    def __len__(self):
        return len(self.batch_idx)
    
    def on_epoch_end(self):
        seed = np.random.randint(5)
        # re-shuffle data for the next epoch
        self.nodes = shuffle(self.nodes, random_state=seed)
        self.edges = shuffle(self.edges, random_state=seed)
        self.y_true = shuffle(self.y_true, random_state=seed)

    def __getitem__(self, idx):
        batch_nodes = self.nodes[self.batch_idx[idx]]
        batch_edges = self.edges[self.batch_idx[idx]]
        features = np.zeros((len(self.batch_idx[idx]), graph_feature_len)) # this is for feature update in the future
        batch_ground_truth = self.y_true[self.batch_idx[idx]]

        return (batch_nodes, batch_edges, features), batch_ground_truth 

# get initial data sets
shuffled_data = data.shuffle(7000, reshuffle_each_iteration=False) # shuffle the data
test_set = shuffled_data.take(1000) # test set: 1000
valid_set = shuffled_data.skip(1000).take(1000) # validation set: 1000
train_set = shuffled_data.skip(2000).take(5000) # training set: 5000

def create_nodes_and_edges(set):
    # nodes
    nodes = [convert_record(d)[0][0] for d in set]
    nodes = [np.repeat(d[np.newaxis, ...], d.shape[0], axis=0) for d in nodes] # reshape from (32, 16) to (32, 32, 16)

    # edges
    coords = [convert_record(d)[0][1] for d in set]
    edges = [coords_to_distances(d) for d in coords]

    return nodes, edges

# preprocess ground truth labels for training set
ys = [convert_record(d)[1] for d in train_set]
train_ym = np.mean(ys)
train_ys = np.std(ys)

# normalizes y
def transform_label(y):
    return (y - train_ym) / train_ys

norm_ys = [transform_label(d) for d in ys]

# create nodes
nodes, edges = create_nodes_and_edges(train_set)

# get the data
batch_size = 32
train_generator = DataGenerator(nodes, edges, norm_ys, batch_size)

# 1. create the model
model = simple_gnn_model()

# 2. compile the model
optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', r2_keras])

# 3. train the model
history = model.fit(train_generator, epochs=100)

# save model
# h5
model.save('gnn_qm9.h5')
# tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('gnn_qm9_model.tflite', 'wb') as f:
  f.write(tflite_model)

# plot metrics
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['mae'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# fig_acc.savefig('gnn_qm9_mae.png', dpi=fig_acc.dpi)

# nodes and edges of validation set
nodes, edges = create_nodes_and_edges(valid_set)

# get ground truth labels in validation set and normalize
ys = [convert_record(d)[1] for d in valid_set]
norm_ys = [transform_label(d) for d in ys]
norm_ys = np.asarray(norm_ys) # convert to np array

features = np.zeros((len(nodes), graph_feature_len))

# 5. evaluate accuracy
accuracy = model.evaluate([np.array(nodes), np.array(edges), features], norm_ys)
print(accuracy)