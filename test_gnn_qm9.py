import tensorflow as tf
from tensorflow import keras
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
    # break up record
    (e, x), y = d
    #
    e = e.numpy()
    x = x.numpy()
    r = x[:, :3]
    # make ohc size larger
    # so use same node feature
    # shape later
    ohc = np.zeros((32, 16))
    ohc[np.arange(len(e)), e - 1] = 1
    #return (ohc, r), y.numpy()[13]
    return (ohc, r), y.numpy()

def coordsToDistance(x):
    #x [N, 3] number of atoms, each atom has (x,y,z) coords
    #x[:, np.newaxis, :] create [N, 1, 3] 
    #(x - x[:, np.newaxis, :]) ** 2), [N, N, 3]
    #for each atom, calculate distances along x,y,z to other atoms(inlcude itself in which distance is 0)
    #sum x,y,z distance [N,N]
    #inverse calculation.
    
    """convert xyz coordinates to inverse pairwise distance"""
    r2 = np.sum((x - x[:, np.newaxis, :]) ** 2, axis=-1)
    e = np.where(r2 != 0.0, 1 / (r2+sys.float_info.epsilon), 0.0)
    ret = np.zeros((32, 32))
    ret[0:x.shape[0], 0:x.shape[0]]=e

    return ret

for d in data:
    (nodes, coords), y = convert_record(d)
    features = np.zeros(graph_feature_len)
    # edges: [N, N] edge between atoms
    edges = coordsToDistance(coords)
    # nodes: [N, 16] expanded to [N, N, 16] for matrix multiplication
    nodes = np.repeat(nodes[np.newaxis, ...], nodes.shape[0], axis=0)

    break

def Conv_block(inputs, feature_dims, activation):
    # Graph Convolution Layer
    outputs = keras.layers.Conv1D(feature_dims, 1, activation=None, use_bias=False)(inputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Activation(activation)(outputs)
    return outputs

def simple_gnn_model():
    # nodes: [N, N, node_feature_len]
    # edges: [N, N]
    nodes = keras.Input((32, 32, node_feature_len))
    edges = keras.Input((32, 32))
    features = keras.Input((graph_feature_len))

    # out: [N, N, msg_feature_len]
    out = Conv_block(nodes, msg_feature_len, 'relu')
    # out: = [N, N, msg_feature_len] . [N, N, 1]

    #out = keras.layers.Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([out, edges])
    x = K.expand_dims(edges, axis=-1)
    x = K.repeat_elements(x, msg_feature_len, axis=-1)
    out = keras.layers.Multiply()([out, x])
    # out: [N, msg_feature_len]
    out = keras.layers.Lambda(lambda x: K.sum(x, axis=2,keepdims=False))(out)
    # new_nodes: [N, node_feature_len]
    new_nodes = Conv_block(out, node_feature_len, 'relu')
    # out: [node_feature_len]
    out = keras.layers.Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(new_nodes)
    # out: [graph_feature_len]
    new_features = keras.layers.Dense(graph_feature_len, use_bias=True)(out)
    new_features = keras.layers.Lambda(lambda x: x[0] + x[1])([new_features, features])
    pred =  keras.layers.Dense(16, use_bias=True)(new_features)

    model = keras.Model(inputs = [nodes, edges, features], outputs = [pred])
    return model


def r2_keras(y_true, y_pred):
    res = keras.backend.sum(keras.backend.square(y_true - y_pred))
    total = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
    return (1-res/(total + keras.backend.epsilon()))

class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_data, y_data, z_data, batch_size):
        self.x, self.y, self.z = np.array(x_data), np.array(y_data), np.array(z_data)
        self.batch_size = batch_size
        self.num_batches = np.ceil(len(x_data)/batch_size)
        self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)

    def __len__(self):
        return len(self.batch_idx)
    
    def on_epoch_end(self):
        seed = np.random.randint(5)
        self.x = shuffle(self.x, random_state=seed)
        self.y = shuffle(self.y, random_state=seed)
        self.z = shuffle(self.z, random_state=seed)

    def __getitem__(self,idx):
        batch_x = self.x[self.batch_idx[idx]]
        batch_y = self.y[self.batch_idx[idx]]
        features = np.zeros((len(self.batch_idx[idx]), graph_feature_len))
        batch_z = self.z[self.batch_idx[idx]]

        return [batch_x, batch_y, features], batch_z 


shuffled_data = data.shuffle(7000, reshuffle_each_iteration=False)
test_set = shuffled_data.take(1000)
valid_set = shuffled_data.skip(1000).take(1000)
train_set = shuffled_data.skip(2000).take(5000)

ys = [convert_record(d)[1] for d in train_set]
train_ym = np.mean(ys)
train_ys = np.std(ys)
print("Mean = ", train_ym, "Std =", train_ys)

def transform_label(y):
    return (y - train_ym) / train_ys

norm_ys = [transform_label(d) for d in ys]

nodes = [(convert_record(d)[0][0]) for d in train_set]
nodes = [np.repeat(d[np.newaxis, ...], d.shape[0], axis=0) for d in nodes]

coords = [(convert_record(d)[0][1]) for d in train_set]
edges = [coordsToDistance(d) for d in coords]

features = np.zeros((len(nodes), graph_feature_len))

model = simple_gnn_model()

optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)

model.compile(loss='mean_squared_error', optimizer = optimizer, metrics=['mae', r2_keras])
print(model.summary())

batch_size = 32
train_generator = DataGenerator(nodes, edges, norm_ys, batch_size)

history = model.fit_generator(generator=train_generator, epochs=100)

model.save('gnn_qm9.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('gnn_qm9_model.tflite', 'wb') as f:
  f.write(tflite_model)
print(model.summary())

fig_acc = plt.figure(figsize=(10,10))
plt.plot(history.history['mae'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
fig_acc.savefig('gnn_qm9_mae.png', dpi=fig_acc.dpi)


nodes = [(convert_record(d)[0][0]) for d in valid_set]
nodes = [np.repeat(d[np.newaxis, ...], d.shape[0], axis=0) for d in nodes]
print((np.array(nodes)).shape)
coords = [(convert_record(d)[0][1]) for d in valid_set]
edges = [coordsToDistance(d) for d in coords]
print((np.array(edges)).shape)

ys = [convert_record(d)[1] for d in valid_set]
norm_ys = [transform_label(d) for d in ys]
norm_ys = np.asarray(norm_ys)
print(norm_ys.shape)

features = np.zeros((len(nodes), graph_feature_len))
print(features.shape)

accuracy = model.evaluate([np.array(nodes), np.array(edges), features], norm_ys)
print(accuracy)