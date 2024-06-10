import tensorflow as tf
import numpy as np
from fetch_data import qm9_parse, qm9_fetch
from tensorflow import keras
import matplotlib.pyplot as plt


qm9_records = qm9_fetch()
data = qm9_parse(qm9_records)

for d in data:
    print(d)
    break

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
    #ohc = np.zeros((len(e), 16))
    ohc = np.zeros((32, 16))
    ohc[np.arange(len(e)), e - 1] = 1
    #return (ohc, r), y.numpy()[13]
    return (ohc, r), y.numpy()


for d in data:
    (e, x), y = convert_record(d)
    print("Element one hots\n", e)
    print("Coordinates\n", x)
    print("Label:", y)
    break


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


def transform_prediction(y):
    return y * train_ys + train_ym

def r2_keras(y_true, y_pred):
    res = keras.backend.sum(keras.backend.square(y_true - y_pred))
    total = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
    return (1-res/(total + keras.backend.epsilon()))

class ReductionLayer(keras.layers.Layer):
    def __init__(self):
        super(ReductionLayer, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=0)

optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)

def simple_keras_model():
    model = keras.Sequential()
    model.add(keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1,keepdims=False)))
    model.add(keras.layers.Dense(16, use_bias=True))
    model.compile(loss='mean_squared_error', optimizer = optimizer, metrics=['mae', r2_keras])
    return model



model = simple_keras_model()

norm_ys = [transform_label(d) for d in ys]
norm_ys = np.asarray(norm_ys)
xs = [(convert_record(d)[0][0]) for d in train_set]
xs_array = np.array(xs)
print(xs_array.shape)
print(norm_ys.shape)

history = model.fit(xs_array, norm_ys, epochs=100, batch_size=32)
model.save('fc_qm9.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('fc_qm9_model.tflite', 'wb') as f:
  f.write(tflite_model)
print(model.summary())

fig_acc = plt.figure(figsize=(10,10))
plt.plot(history.history['mae'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
fig_acc.savefig('fc_qm9_mae.png', dpi=fig_acc.dpi)

xs = [(convert_record(d)[0][0]) for d in valid_set]
xs_array = np.array(xs)
ys = [convert_record(d)[1] for d in valid_set]
norm_ys = [transform_label(d) for d in ys]
norm_ys = np.asarray(norm_ys)

accuracy = model.evaluate(xs_array, norm_ys)
print(accuracy)






