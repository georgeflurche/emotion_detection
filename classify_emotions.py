'''
This module is used to train a Deep Neural Network that can recognize emotions
using EEG time-series
'''
import os
import sys
import argparse
import logging
import json
import csv
import re
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from enum import Enum, unique
import random as rnd
import seaborn
import matplotlib.pyplot as plt
from data_interact import *


logging.basicConfig(
    format='%(levelname)s [%(module)s]: %(message)s'
)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

@unique
class Label(Enum):
    HAPPINESS = 0
    SADNESS = 1
    FEAR = 2


class TrainConfig:
    def __init__(self, config_file=None, **kwargs):
        self.config_file = config_file
        if self.config_file is not None:
            # Set the data from config, then overwrite the attributes from the
            # code if there are any
            self._set_config_from_file()
        self.set_attributes(**kwargs)

    def set_attributes(self, **kwargs):
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    def _set_config_from_file(self):
        config_abspath = os.path.expandvars(self.config_file)
        if os.path.isfile(config_abspath) and config_abspath.endswith('.json'):
            try:
                with open(config_abspath) as json_file:
                    config = json.load(json_file)
            except Exception as e:
                _logger.error(
                    f"The file {self.config_file} is not a valid json\n{e}")
                sys.exit(1)
            for k, v in config.items():
                # Set all of the attributes according to the config file
                setattr(self, k, v)
        else:
            _logger.error(f"The file {self.config_file} does not exist")
            sys.exit(1)

    def _extract_data(self):
        data = list()
        dataset_path = os.path.expandvars(self.dataset_path)
        if os.path.isfile(dataset_path) and dataset_path.endswith('.csv'):
            try:
                with open(dataset_path, newline="\n") as csvfile:
                    reader = csv.reader(csvfile, delimiter=",",quotechar='"')
                    for row in reader:
                        data.append(row)
            except Exception as e:
                _logger.error(f"Could not load data from csv file "
                              f"{self.dataset_path}. Exception occured:\n{e}")
                sys.exit(1)
        else:
            _logger.error(
                f"The file {self.dataset_path} doesn't exist or it's not a csv")
            sys.exit(1)
        numeric_data = list()
        labels = list()
        features_list = data[0]
        for row in data[1:]:
            numeric_data.append(list(float(i) for i in row[:-1]))
            labels.append(row[-1])
        return numeric_data, labels, features_list

    def extract_network_data(self):
        numeric_data, labels, features_list = self._extract_data()
        if self.dnn_type == "CNN":
            channels = None
            X = list()
            y = list()
            for j, row in enumerate(numeric_data):
                row_channels = dict()
                for i, feature in enumerate(features_list[:-1]):
                    index = re.search(r'\d+_(?=[ab])', feature).group(0)
                    feature_kind = feature.replace(index, '')
                    if feature_kind in row_channels:
                        row_channels[feature_kind].append(row[i])
                    else:
                        row_channels[feature_kind] = [row[i]]
                if channels is None:
                    channels = sorted(row_channels.keys())
                dim3 = list()
                for ch in channels:
                    dim3.append(row_channels[ch])
                X.append(dim3)
                if labels[j] == 'HAPPY':
                    y.append(Label.HAPPINESS)
                elif labels[j] == 'SAD':
                    y.append(Label.SADNESS)
                elif labels[j] == 'FEAR':
                    y.append(Label.FEAR)
                else:
                    _logger.error(f"Unrecognized label {labels[j]}")
                    sys.exit(1)
            training_length = int(self.training_weight*len(X))
            testing_length = len(X) - training_length
            return (X[:training_length], y[:training_length],
                    X[training_length:], y[training_length: ], channels)
        else:
            _logger.error(f"Unrecognized DNN Type {self.dnn_type}")
            sys.exit(1)

    def extract_filtered_network_data(self):
        dir_path = os.path.expandvars(self.dataset_path)
        all_samples = sorted(list(
            i for i in os.listdir(dir_path) if 'filtered_sample_' in i))
        X = list()
        for sample in all_samples:
            data_path = os.path.join(dir_path, sample)
            with open(data_path) as f:
                data_matrix = list(list(float(i) for i in row.split(','))
                                   for row in f.read().split('\n'))
            data_arr = np.array(data_matrix)
            data_arr = np.transpose(data_arr)
            data_arr = data_arr[:4]
            if self.normalize:
                scaler = np.array(list(max(abs(np.max(i)), abs(np.min(i))) for i in data_arr)).reshape(4, 1)
                data_arr = data_arr / scaler
            if self.nof_dimensions == 2:
                data_arr = np.array(list(np.concatenate((
                    data_arr[0], data_arr[1], data_arr[2], data_arr[3]
                    ), axis=None)))
            X.append(data_arr)
        X = np.array(X)
        with open(os.path.join(dir_path, 'labels_map.csv')) as f:
            lbls = list(row.split(',')[-1] for row in f.read().split('\n')[1:])
        y = list()
        for lbl in lbls:
            if lbl == 'HAPPINESS':
                y.append(Label.HAPPINESS)
            elif lbl == 'SADNESS':
                y.append(Label.SADNESS)
            elif lbl == 'FEAR':
                y.append(Label.FEAR)
            else:
                _logger.error(f"Unrecognized label {lbl}")
                sys.exit(1)
        y = np.array(list(i.value for i in y))
        y = y.reshape(len(y), 1)

        training_lines = int(self.training_weight * len(all_samples))

        if self.nof_dimensions == 2:
            full_data = np.concatenate((X, y), axis=1)

            # Shuffle the data so the patients to be different
            np.random.seed(self.seed)
            np.random.shuffle(full_data)
            train_data = full_data[:training_lines]
            test_data = full_data[training_lines:]
            X_train = train_data[:, : -1]
            X_test = test_data[:, : -1]
            y_train = train_data[:, -1]
            y_test = test_data[:, -1]
        else:
            X_train = X[: training_lines]
            X_test = X[training_lines:]
            y_train = y[:training_lines]
            y_test = y[training_lines:]


        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)

        return X_train, y_train, X_test, y_test

    def plot_random_data(self, X, y, channels):
        row_index = rnd.randint(0, len(X)-1)
        row = X[row_index]
        label = y[row_index].name
        fig, axs = plt.subplots(9, 2)
        fig.suptitle(f"Channels for label {label}")
        j = -1
        for i, channel in enumerate(channels):
            if len(row[i]) <= 5:
                continue
            else:
                j += 1
            if j%2 == 0:
                axs[j//2, 0].plot(row[i])
                axs[j//2, 0].set_title(f"Channel {j//2} A")
            else:
                axs[j//2, 1].plot(row[i])
                axs[j//2, 1].set_title(f"Channel {j//2} B")
        plt.show()

    def flatten_data(self, X_train, y_train, X_test, y_test):
        new_X_train = list()
        new_X_test = list()
        new_y_train = list()
        new_y_test = list()
        # Prepare training data
        for row in X_train:
            new_row = list()
            for ch in row:
                if len(ch) > 20:
                    new_row += ch
            new_X_train.append(new_row)
        new_X_train = np.array(new_X_train)
        l, w = new_X_train.shape
        new_X_train = new_X_train.reshape(l, 2, w//2, order='F')
        new_y_train = np.array(list(i.value for i in y_train)).reshape(l, 1)
        # Prepare testing datasets
        for row in X_test:
            new_row = list()
            for ch in row:
                if len(ch) > 20:
                    new_row += ch
            new_X_test.append(new_row)
        new_X_test = np.array(new_X_test)
        l, w = new_X_test.shape
        new_X_test = new_X_test.reshape(l, 2, w//2, order='F')

        new_y_test = np.array(list(i.value for i in y_test)).reshape(l, 1)

        #Vectorize data
        if self.nof_dimensions == 2:
            _logger.info('Vectorizing data do 2 dimensions')
            ftrain = list()
            for d, i in enumerate(new_X_train):
                ftrain.append(list(np.concatenate((i[0], i[1]), axis=None)))
            new_X_train = np.array(ftrain)

            ftest = list()
            for d, i in enumerate(new_X_test):
                ftest.append(list(np.concatenate((i[0], i[1]), axis=None)))
            new_X_test = np.array(ftest)
        elif self.nof_dimensions == 3:
            _logger.info(f'The data has 3 dimensions: {new_X_test.shape}')

        return new_X_train, new_y_train, new_X_test, new_y_test

    def get_architecture(self, input_shape):
        architecture = models.Sequential()
        if config.dnn_type == 'CNN':
            if config.nof_dimensions == 2 and config.nof_conv_layers == 1:
                architecture.add(layers.Conv1D(
                    filters=32, kernel_size=3, activation='relu',
                    input_shape=input_shape))
                architecture.add(layers.MaxPooling1D(pool_size=20))
                architecture.add(layers.Flatten())
                #architecture.add(layers.Dense(100, activation='relu'))
                architecture.add(layers.Dense(3, activation='softmax'))

            elif config.nof_dimensions == 2 and config.nof_conv_layers == 2:
                architecture.add(layers.Conv1D(
                    filters=32, kernel_size=3, activation='relu',
                    input_shape=input_shape))
                architecture.add(layers.MaxPooling1D(pool_size=20))
                architecture.add(layers.Conv1D(
                    filters=64, kernel_size=6, activation='relu'))
                architecture.add(layers.MaxPooling1D(pool_size=20))
                architecture.add(layers.Dropout(0.25))
                architecture.add(layers.Flatten())
                architecture.add(layers.Dense(3, activation='softmax'))

            elif config.nof_dimensions == 2 and config.nof_conv_layers == 3:
                architecture.add(layers.Conv1D(
                    filters=32, kernel_size=3, activation='relu',
                    input_shape=input_shape))
                architecture.add(layers.MaxPooling1D(pool_size=20))
                architecture.add(layers.Conv1D(
                    filters=64, kernel_size=6, activation='relu'))
                architecture.add(layers.MaxPooling1D(pool_size=20))
                architecture.add(layers.Conv1D(
                    filters=64, kernel_size=3, activation='relu'))
                architecture.add(layers.Dropout(0.35))
                architecture.add(layers.Flatten())
                architecture.add(layers.Dense(3, activation='softmax'))

            elif config.nof_dimensions == 3 and config.nof_conv_layers == 1:
                 architecture.add(layers.Conv2D(
                     filters=32, kernel_size=3, activation='relu',
                     input_shape=input_shape))
                 architecture.add(layers.MaxPooling2D(pool_size=2))
                 architecture.add(layers.Dropout(0.35))
                 architecture.add(layers.Flatten())
                 architecture.add(layers.Dense(3, activation='softmax'))
            else:
                _logger.error(
                    f'Unimplemented architecture for {config.nof_dimensions} '
                    f'dimensions and {config.nof_conv_layers} layers: CNN')
                sys.exit(1)
        else:
            if config.nof_dimensions == 2:
                architecture.add(layers.InputLayer(input_shape=input_shape))
                architecture.add(layers.GRU(256, return_sequences=True))
                architecture.add(layers.Flatten())
                architecture.add(layers.Dense(3))
            else:
                _logger.error(
                    f'Unimplemented architecture for {config.nof_dimensions} '
                    f'dimensions and {config.nof_conv_layers} layers: RNN')
                sys.exit(1)
        return architecture


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", type=str, default=None, help=(
        "Path to the JSON config file"))
    args = parser.parse_args()

    _logger.info("Generationg the config")
    config = TrainConfig(args.config_file)
    _logger.info("Extracting network data")
    X_train, y_train, X_test, y_test= config.extract_filtered_network_data()

    if config.dnn_type == 'CNN':
        if config.nof_dimensions == 2:
            x_tr_l, x_tr_w = X_train.shape
            x_ts_l, x_ts_w = X_test.shape
            input_shape = (x_tr_w, 1)
            X_train = X_train.reshape(x_tr_l, x_tr_w, 1)
            X_test = X_test.reshape(x_ts_l, x_ts_w, 1)
        else:
            x_tr_l, x_tr_w, x_tr_d = X_train.shape
            x_ts_l, x_ts_w, x_ts_d = X_test.shape
            input_shape = (x_tr_w, x_tr_d, 1)
            X_train = X_train.reshape(x_tr_l, x_tr_w, x_tr_d, 1)
            X_test = X_test.reshape(x_ts_l, x_ts_w, x_ts_d, 1)
    else:
         if config.nof_dimensions == 2:
             x_tr_l, x_tr_w = X_train.shape
             x_ts_l, x_ts_w = X_test.shape
             input_shape = (x_tr_w, 1)
             X_train = X_train.reshape(x_tr_l, x_tr_w, 1)
             X_test = X_test.reshape(x_ts_l, x_ts_w, 1)
         else:
             input_shape = X_train.shape[1:]


    # Change the output vector into binary class matrix
    y_train = tf.keras.utils.to_categorical(
        y_train, num_classes=3, dtype='float32')
    y_test = tf.keras.utils.to_categorical(
        y_test, num_classes=3, dtype='float32')

    dstr= datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        f"logs/fit/{config.dnn_type}_{config.nof_dimensions}D_"
        f"{config.nof_conv_layers}_layers_{config.training_epochs}_epochs_" +
        dstr)

    # Define callbacks for tensorflow
    callbacks = list()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir)
    callbacks.append(tensorboard_callback)
    if config.early_stop is not None:
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stop,
            restore_best_weights=True
        )
        callbacks.append(earlystop_callback)

    # Define the model
    model = config.get_architecture(input_shape)
    model.summary()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0,
            reduction="auto",
            name="categorical_crossentropy"),
        metrics=['accuracy'])

    # Train the model according to the config
    if config.cross_validate:
        inputs = np.concatenate((X_train, X_test), axis=0)
        targets = np.concatenate((y_train, y_test), axis=0)
        kfold = KFold(n_splits=config.num_folds, shuffle=True)
        ecoef = config.training_epochs//config.num_folds

        fold_no = 0
        for train, test in kfold.split(inputs, targets):
            initial_epoch = ecoef*fold_no + 1
            epochs = ecoef * (fold_no+1)

            _logger.info(f'Training for fold {fold_no+1} ...')
            history = model.fit(
                inputs[train], targets[train], initial_epoch=initial_epoch,
                epochs=epochs, validation_data=(inputs[test], targets[test]),
                callbacks=callbacks)
            fold_no += 1

        scores = model.evaluate(X_test, y_test, verbose=0)
        _logger.info(f'The final score of the model is:\n '
                     f'{model.metrics_names[0]}: {round(scores[0], 4)}\n'
                     f'{model.metrics_names[1]}: {round(scores[1], 4)}')

    else:
        history = model.fit(
            X_train, y_train, epochs=config.training_epochs,
            validation_data=(X_test, y_test), callbacks=callbacks,
            batch_size=64)

    if config.show_confusion_matrix:
        model_predictions = np.argmax(model.predict(X_test), axis=1)
        y_test = np.argmax(y_test, axis=1)
        conf_mat = confusion_matrix(y_test, model_predictions)
        plt.figure()
        seaborn.heatmap(conf_mat, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
        plt.xticks(ticks=list(i+0.5 for i in range(3)),
                   labels=list(i.name for i in Label))
        plt.yticks(ticks=list(i+0.5 for i in range(3)),
                   labels=list(i.name for i in Label))
        plt.xlabel('Predicted Values')
        plt.ylabel('Real Values')
        plt.show()
