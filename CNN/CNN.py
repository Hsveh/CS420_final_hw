import tensorflow as tf
import time
import datetime
import os
1
import common
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
exp_v = "1.4.1"
batch_size = 1500
fig = 28
max_iter = 300000
log_dir = "log/" + exp_v
model_dir = "model/" + exp_v
model_save_iter = 5000
if os.path.exists(log_dir):
    os.remove(log_dir)
if os.path.exists(model_dir):
    os.remove(model_dir)
os.makedirs(log_dir)
os.makedirs(model_dir)
starttime = datetime.datetime.now()

data = common.Data("../mnist/mnist_train/train_data.npy", "../mnist/mnist_train/mnist_train_label", "../mnist/mnist_test/test_data.npy", "../mnist/mnist_test/mnist_test_label", batch_size, 28)


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3x3(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, fig*fig], name="input")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    W_conv1 = weight_variable([2, 2, 1, 32], name="W_conv1")
    b_conv1 = bias_variable([32], name="b_conv1")
    x_image = tf.reshape(x, [-1, fig, fig, 1], name="x_image")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="h_conv1")
    h_pool1 = max_pool_3x3(h_conv1, name="h_pool1")
    print(h_pool1.shape)
    W_conv2 = weight_variable([2, 2, 32, 64], name="W_conv2")
    b_conv2 = bias_variable([64], name="b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="h_conv2")
    h_pool2 = max_pool_3x3(h_conv2, name="h_pool2")
    print(h_pool2.shape)
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W_fc1")
    b_fc1 = bias_variable([1024], name="b_fc1")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name="h_pool2_flat")
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10], name="W_fc2")
    b_fc2 = bias_variable([10], name="b_fc2")
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), name="cross_entropy")
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    train_writer = tf.summary.FileWriter((log_dir+"/train"), graph)
    test_writer = tf.summary.FileWriter((log_dir+"/test"), graph)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables())
    tf.add_to_collection('pred_network', y_conv)
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    starttime_ = datetime.datetime.now()
    for i in range(max_iter):
        batch_x, batch_y = data.next_batch()
        _ = sess.run([train_step], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
        if i % 100 == 0:
            # acc = sess.run([accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            train_writer.add_summary(summary, i)
            summary, t_acc = sess.run([merged, accuracy], feed_dict={x: data.test_x, y_: data.test_y, keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            endtime_ = datetime.datetime.now()
            print(endtime_ - starttime_)
            print('step: {}, accuracy: {}, test_acc: {}'.format(i, acc, t_acc))
            starttime_ = datetime.datetime.now()
        if i % model_save_iter == 0:
            saver.save(sess, model_dir + '/' + str(i) + '/' + str(i))
    acc = sess.run([accuracy], feed_dict={x: data.test_x, y_: data.test_y, keep_prob: 1.0})
    if i == max_iter-1:
        saver.save(sess, model_dir + '/' + str(max_iter) + '/' + str(max_iter))
endtime = datetime.datetime.now()
print(endtime - starttime)
print("ACC:")
print(acc)
