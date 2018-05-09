import numpy as np
from keras import backend as K


def readData(train_data_file, train_label_file, test_data_file, test_label_file, fig_w):
    train_x = np.fromfile(train_data_file, dtype=np.uint8)
    train_y = np.fromfile(train_label_file, dtype=np.uint8)
    train_data_num = len(train_y)
    train_x = train_x.reshape(train_data_num,fig_w*fig_w)
    test_x = np.fromfile(test_data_file, dtype=np.uint8)
    test_y = np.fromfile(test_label_file, dtype=np.uint8)
    test_data_num = len(test_y)
    test_x = test_x.reshape(test_data_num, fig_w*fig_w)
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.int32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.int32)
    train_y_ = np.zeros(shape=(train_data_num, 10))
    test_y_ = np.zeros(shape=(test_data_num, 10))
    for i in range(train_data_num):
        train_y_[i, train_y[i]] = 1
    for i in range(test_data_num):
        test_y_[i, test_y[i]] = 1
    print('train size:%s, test size:%s' % (train_data_num, test_data_num))
    return train_x, train_y, test_x, test_y


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
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

