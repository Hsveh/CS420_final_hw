import numpy as np
from keras import backend as K
import logging
import os


class Data():
    def __init__(self, train_x, train_y, test_x, test_y, batch_size, fig_w):
        self.train_x_path = train_x
        self.train_y_path = train_y
        self.test_x_path = test_x
        self.test_y_path = test_y
        self.batch_size = batch_size
        self.fig_w = fig_w
        self.ptr = 0
        self.size = 0
        self.size_test = 0
        self.train_x, self.train_y, self.test_x, self.test_y = self.read_data()
        if self.batch_size > self.size:
            return -1

    def read_data(self):
        train_x = np.fromfile(self.train_x_path, dtype=np.uint8)
        train_y = np.fromfile(self.train_y_path, dtype=np.uint8)
        self.size = len(train_y)
        train_x = train_x.reshape(self.size, self.fig_w**2)
        test_x = np.fromfile(self.test_x_path, dtype=np.uint8)
        test_y = np.fromfile(self.test_y_path, dtype=np.uint8)
        self.size_test = len(test_y)
        test_x = test_x.reshape(self.size_test, self.fig_w**2)
        train_x = train_x.astype(np.float32)
        train_y = train_y.astype(np.int32)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.int32)
        train_y_ = np.zeros(shape=(self.size, 10))
        test_y_ = np.zeros(shape=(self.size_test, 10))
        for i in range(self.size):
            train_y_[i, train_y[i]] = 1
        for i in range(self.size_test):
            test_y_[i, test_y[i]] = 1
        return train_x, train_y_, test_x, test_y_

    def next_batch(self):
        if self.ptr + self.batch_size >= self.size:
            head = 0
            tail = self.batch_size
            self.ptr = self.batch_size
        else:
            head = self.ptr
            tail = self.ptr + self.batch_size
            self.ptr += self.batch_size
        return self.train_x[head:tail, 0:self.fig_w**2], self.train_y[head:tail, 0:10]


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_ = true_positives / (possible_positives + K.epsilon())
    return recall_


def create_logger(logger_name,
                  log_format=None,
                  log_level=logging.INFO,
                  log_path=None):
    logger = logging.getLogger(logger_name)
    assert (len(logger.handlers) == 0)
    logger.setLevel(log_level)
    if log_path is None:
        handler = logging.StreamHandler()
    else:
        os.stat(os.path.dirname(os.path.abspath(log_path)))
        handler = logging.FileHandler(log_path)
    handler.setLevel(log_level)
    if log_format is not None:
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
