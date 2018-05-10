from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import common as c
import pprint
import os

sys.path.append('../')

pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_string("gpu", "0", "GPU(s) to use. [0]")
flags.DEFINE_integer("save_step", 500, "The interval of saving checkpoints[500]")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log", "summary", "log [log]")
flags.DEFINE_integer("epoch", 100, "Epoch[10]")
flags.DEFINE_string("model_dir", "model", "Model Path")
flags.DEFINE_string("train_data_file", "../../mnist/mnist_train/mnist_train_data", "Train data file")
flags.DEFINE_string("train_label_file", "../../mnist/mnist_train/mnist_train_label", "Train label file")
flags.DEFINE_string("test_data_file", "../../mnist/mnist_test/mnist_test_data", "Test data file")
flags.DEFINE_string("test_label_file", "../../mnist/mnist_test/mnist_test_label", "Test label file")
flags.DEFINE_integer("fig_w", 45, "Image Size")
flags.DEFINE_integer("batch_size", 100, "Batch Size")
flags.DEFINE_integer("epochs", 100, "Epoch")
flags.DEFINE_float("learning_rate", 0.001, "Learning Rate")
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def cnn(features, labels, mode):
    # [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 45, 45, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    # in 45*45 32 out 15*15 64
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    # in 15*15 64 out 5*5 64
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)

    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    sys.stdout = os.fdopen(sys.__stdout__.fileno(), 'w', 0)
    if not os.path.isdir(FLAGS.checkpoint):
        os.mkdir(FLAGS.checkpoint)
    if not os.path.isdir(FLAGS.log):
        os.mkdir(FLAGS.log)

    train_x, train_y, test_x, test_y = c.readData(FLAGS.train_data_file, FLAGS.train_label_file, FLAGS.test_data_file,
                                                FLAGS.test_label_file, FLAGS.fig_w)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn, model_dir=FLAGS.model_dir)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=7000)

    # Evaluate the model and print results
    test_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_x},
        y=test_y,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=test_input)
    print(eval_results)

if __name__ == '__main__':
    tf.app.run()