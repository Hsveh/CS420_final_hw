from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import pprint
import os
import sys
sys.path.append('../')
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout,Input
from keras.utils import to_categorical
from keras import regularizers,initializers
from keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from common import readData, recall, precision, f1

pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_string("gpu", "0", "GPU(s) to use. [0]")
flags.DEFINE_float("learning_rate", 2.5e-3, "Learning rate [2.5e-4]")
flags.DEFINE_integer("batch_size", 200, "The number of batch images [4]")
flags.DEFINE_integer("save_step", 500, "The interval of saving checkpoints[500]")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log", "summary", "log [log]")
flags.DEFINE_integer("epoch", 100, "Epoch[10]")
flags.DEFINE_string("train_data_file", "../../mnist/mnist_train/mnist_train_data", "Train data file")
flags.DEFINE_string("train_label_file", "../../mnist/mnist_train/mnist_train_label", "Train label file")
flags.DEFINE_string("test_data_file", "../../mnist/mnist_test/mnist_test_data", "Test data file")
flags.DEFINE_string("test_label_file", "../../mnist/mnist_test/mnist_test_label", "Test label file")
FLAGS = flags.FLAGS

train_data_num = 60000  # The number of figures
test_data_num = 10000
fig_w = 45  # width of each figure

def regularizer(model,kernel_regularizer = regularizers.l2(),bias_regularizer=regularizers.l2()):
    for layer in model.layers:
        if hasattr(layer,"kernel_regularizer"):
            layer.kernel_regularizer = kernel_regularizer
        if hasattr(layer, "bias_regularizer"):
            layer.bias_regularizer = bias_regularizer

def genModel(neural,features=200):
    inputs = Input(shape=(features,))
    hidden1 = Dropout(0.5)(BatchNormalization(axis=1)(Dense(neural[0], activation='softmax')(inputs)))
    hidden2 = Dropout(0.5)(BatchNormalization(axis=1)(Dense(neural[1], activation='softmax')(hidden1)))
    #hidden3 = Dropout(0.5)(BatchNormalization(axis=1)(Dense(neural[2], activation='relu')(hidden2)))
    outputs = Dense(193,activation='softmax')(hidden2)
    model = Model(inputs = inputs, outputs=outputs)
    regularizer(model)
    return model

def main():
    pp.pprint(flags.FLAGS.__flags)
    sys.stdout = os.fdopen(sys.__stdout__.fileno(), 'w', 0)
    if not os.path.isdir(FLAGS.checkpoint):
        os.mkdir(FLAGS.checkpoint)
    if not os.path.isdir(FLAGS.log):
        os.mkdir(FLAGS.log)

    train_x, train_y, test_x, test_y=readData(FLAGS.train_data_file, FLAGS.train_label_file, FLAGS.test_data_file, FLAGS.test_label_file, fig_w)

    model = genModel([1000, 1500])
    model.summary()

    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', recall, precision, f1])

    train_y = to_categorical(train_y, num_classes=10)
    test_y = to_categorical(test_y, num_classes=10)

    model_path = os.path.join(FLAGS.checkpoint, "weight.hdf5")
    callbacks = [
        ModelCheckpoint(filepath=model_path, monitor='val_acc', save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir=FLAGS.log),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2)
    ]
    hist = model.fit(train_x, train_y, epochs=FLAGS.epoch, batch_size=100, validation_data=(test_x, test_y),
                     callbacks=callbacks)
    loss, accuracy, re, pre, f1_s = model.evaluate(test_x, test_y, batch_size=100, verbose=1)
    print(hist)

if __name__ == '__main__':
    # tf.app.run()
    main()