import tensorflow as tf
import numpy as np
from fetch_data import qm9_parse, qm9_fetch
from tensorflow import keras
import matplotlib.pyplot as plt

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

# same as tf.keras.metrics.R2Score but as a function
def r2_keras(y_true, y_pred):
    res = keras.backend.sum(keras.backend.square(y_true - y_pred))
    total = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
    return (1 - res/(total + keras.backend.epsilon()))

# get data
shuffled_data = data.shuffle(7000, reshuffle_each_iteration=False)
test_set = shuffled_data.take(1000)
valid_set = shuffled_data.skip(1000).take(1000)
train_set = shuffled_data.skip(2000).take(5000)

ys = [convert_record(d)[1] for d in train_set]
train_ym = np.mean(ys)
train_ys = np.std(ys)

# normalize ground truth using mean and standard deviation
def transform_label(y):
    return (y - train_ym) / train_ys

# un-normalize ground truth (basically, reverse of transform_label)
def transform_prediction(y):
    return y * train_ys + train_ym

class ReductionLayer(keras.layers.Layer):
    def __init__(self):
        super(ReductionLayer, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=0)

def simple_keras_model():
    model = keras.Sequential()
    model.add(keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1,keepdims=False)))
    model.add(keras.layers.Dense(16, use_bias=True))
    return model

# get data
# get one-hot-encoded elements
xs = [convert_record(d)[0][0] for d in train_set]
xs_array = np.array(xs)

# get normalized ground truths
norm_ys = [transform_label(d) for d in ys]
norm_ys = np.asarray(norm_ys)

# 1. create model
model = simple_keras_model()

# 2. compile model
optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', r2_keras])

# 3. fit model
history = model.fit(xs_array, norm_ys, epochs=100, batch_size=32)

# save the model
# h5
model.save('fc_qm9.h5')
# tflite
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open('fc_qm9_model.tflite', 'wb') as f:
#   f.write(tflite_model)

# plot metrics
fig_acc = plt.figure(figsize=(10,10))
plt.plot(history.history['mae'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# fig_acc.savefig('fc_qm9_mae.png', dpi=fig_acc.dpi)

# 4. get validation data
# get validation one-hot-encoded elements
xs = [(convert_record(d)[0][0]) for d in valid_set]
xs_array = np.array(xs)
# get normalized validation ground truths
ys = [convert_record(d)[1] for d in valid_set]
norm_ys = [transform_label(d) for d in ys]
norm_ys = np.asarray(norm_ys)

# 5. evaluate validation accuracy
accuracy = model.evaluate(xs_array, norm_ys)
print(accuracy)






