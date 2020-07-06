import time

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dropout, Dense


class SentimentAnalyzer:
    BATCH_SIZE = 500
    TRAIN_DATA_SIZE = 10000
    train_data = None
    validation_data = None
    test_data = None
    model = None
    encoder = None

    def initialize_gpus(self):
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

    def load_encodedimdb_dataset(self):
        dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
        self.train_data, self.test_data = dataset['train'], dataset['test']
        self.encoder = info.features['text'].encoder
        return self.train_data, self.validation_data

    def load_imdb_dataset(self):
        self.train_data, self.validation_data, self.test_data = tfds.load(
            name="imdb_reviews",
            split=('train[:60%]', 'train[60%:]', 'test'),
            as_supervised=True)
        return self.train_data, self.validation_data, self.test_data

    def train_model(self, lstm_layer, dense_layer, layer_size, save=False):
        MODEL_NAME = "%d-lstm%d-nodes%d-dense_%d" % (lstm_layer, layer_size, dense_layer, (time.time()))
        tensorboard = TensorBoard(log_dir='tensorboard_logs\\{}'.format(MODEL_NAME))

        embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
        embedding_layer = hub.KerasLayer(embedding, input_shape=[],
                                         dtype=tf.string, trainable=True)

        model = tf.keras.Sequential()
        model.add(embedding_layer)
        model.add(Dropout(0.2))

        model.add(Conv1D(filters=layer_size, kernel_size=2, padding="valid", activation="relu", strides=1))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Bidirectional(LSTM(layer_size // 2, recurrent_dropout=0.2)))
        model.add(Dropout(0.2))

        for i in range(lstm_layer - 1):
            model.add(Bidirectional(LSTM(layer_size // 2, recurrent_dropout=0.2)))
            model.add(Dropout(0.2))

        model.add(Dense(layer_size))
        model.add(Dropout(0.2))

        for i in range(dense_layer - 1):
            model.add(Dense(layer_size))
            model.add(Dropout(0.2))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

        model.fit(self.train_data.shuffle(self.TRAIN_DATA_SIZE).batch(self.BATCH_SIZE),
                  epochs=20,
                  validation_data=self.validation_data.batch(self.BATCH_SIZE),
                  verbose=1, callbacks=[tensorboard])
        if save:
            model.save(MODEL_NAME)
        return model

    def optimize_model(self):
        self.initialize_gpus()
        self.load_imdb_dataset()
        dense_layers = [1, 2, 3]
        lstm_layers = [1, 2]
        layer_sizes = [16, 32, 64]
        for dense in dense_layers:
            for lstm in lstm_layers:
                for size in layer_sizes:
                    self.train_model(lstm, dense, size, True)

    def predict_sample(self, sample):
        return self.model.predict([sample])

    def get_max_and_min_predictions(self):
        examples, _ = next(iter(self.test_data.batch(25000)))
        min = [999]
        max = [-999]
        prediction = self.model.predict(examples)
        for idx, i in enumerate(prediction):
            if i[0] < min[0]:
                min = [i[0], idx]
            if i[0] > max[0]:
                max = [i[0], idx]
        min[1] = tf.gather_nd(examples, tf.stack([min[1]], -1)).numpy()
        max[1] = tf.gather_nd(examples, tf.stack([max[1]], -1)).numpy()
        print("max = ", max, "\nmin = ", min)

    def init_module(self, model_name=None):
        self.initialize_gpus()
        self.load_encodedimdb_dataset()
        self.load_imdb_dataset()
        if model_name is None:
            model_name = "../processing/sentiment_analysis/models/1-lstm_32-nodes_2-dense_1585211592.h5"
        self.model = tf.keras.models.load_model(model_name)
        print(self.model.summary())

    def pad_to_size(self, vec, size):
        zeros = [0] * (size - len(vec))
        vec.extend(zeros)
        return vec

    def pad_predict_sample(self, sample, pad):
        encoded_sample_pred_text = self.encoder.encode(sample)
        num_of_words = len(encoded_sample_pred_text)
        if pad:
            encoded_sample_pred_text = self.pad_to_size(encoded_sample_pred_text, 256)
        encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
        predictions = self.model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
        if pad:
            predictions = predictions[0, :num_of_words, 0]
        sum = 0.0
        num = 0.0
        for x in predictions:
            sum += float(x)
            num += 1
        return sum / num


if __name__ == "__main__":
    sentimentAnalyzer = SentimentAnalyzer()
    sentimentAnalyzer.init_module()
    print(sentimentAnalyzer.predict_sample(
        "I am pleasantly surprised by the amount of cameras. The bezel is a little too big."))
