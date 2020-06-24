import time

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense

BATCH_SIZE = 500
TRAIN_DATA_SIZE = 10000
train_data = None
validation_data = None
test_data = None
model = None
encoder = None


def initialize_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print('Error: ', e)
    else:
        print('No GPUs found')

def load_encodedimdb_dataset():
    global train_data, validation_data, encoder
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_data, test_data = dataset['train'], dataset['test']
    encoder = info.features['text'].encoder
    return train_data, validation_data


def load_imdb_dataset():
    global train_data, validation_data, test_data
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)
    return train_data, validation_data, test_data


def train_model(dense_layer, dropout_layer, layer_size, save=False):
    global train_data, validation_data, test_data
    MODEL_NAME = "%d-dense_%d-dropout_%d-size_%d" % (dense_layer, dropout_layer, layer_size, (time.time()))
    tensorboard = TensorBoard(log_dir='tensorboard_logs\\{}'.format(MODEL_NAME))

    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    embedding_layer = hub.KerasLayer(embedding, input_shape=[],
                                     dtype=tf.string, trainable=True)

    model = tf.keras.Sequential()
    model.add(embedding_layer)
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

    for i in range(dense_layer - 1):
        model.add(Dense(layer_size))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_data.shuffle(TRAIN_DATA_SIZE).batch(BATCH_SIZE),
                        epochs=20,
                        validation_data=validation_data.batch(BATCH_SIZE),
                        verbose=1, callbacks=[tensorboard])
    if save:
        model.save(MODEL_NAME)
    return model


def predict_sample(sample):
    return model.predict([sample])


def get_max_and_min_predictions():
    examples, _ = next(iter(test_data.batch(25000)))
    min = [999]
    max = [-999]
    prediction = model.predict(examples)
    for idx, i in enumerate(prediction):
        if i[0] < min[0]:
            min = [i[0], idx]
        if i[0] > max[0]:
            max = [i[0], idx]
    min[1] = tf.gather_nd(examples, tf.stack([min[1]], -1)).numpy()
    max[1] = tf.gather_nd(examples, tf.stack([max[1]], -1)).numpy()
    print("max = ", max, "\nmin = ", min)


def init_module(model_name=None):
    global model
    initialize_gpus()
    load_encodedimdb_dataset()
    load_imdb_dataset()
    if model_name is None:
        model_name = "../processing/sentiment_analysis/models/1-lstm_32-nodes_2-dense_1585211592.h5"
    model = tf.keras.models.load_model(model_name)
    print(model.summary())


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def pad_predict_sample(sample, pad):
    global encoder
    encoded_sample_pred_text = encoder.encode(sample)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 256)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    sum = 0.0
    num = 0.0
    for x in predictions[0]:
        sum += float(x[0])
        num += 1
    return sum / num


def optimize_model():
    initialize_gpus()
    load_imdb_dataset()
    dense_layers = [1, 2, 3]
    dropout_layers = [0, 1]
    layer_sizes = [16, 32, 64]
    for dense in dense_layers:
        for dropout in dropout_layers:
            for size in layer_sizes:
                train_model(dense, dropout, size, True)


if __name__ == "__main__":
    init_module()
    print(predict_sample("I am pleasantly surprised by the amount of cameras. The bezel is a little too big."))
