import tensorflow as tf
import os
import numpy as np
from PIL import Image
import sys
sys.path.append("../")
import common
data = common.Data("../mnist/mnist_train/mnist_train_data", "../mnist/mnist_train/mnist_train_label", "../mnist/mnist_test/mnist_test_data", "../mnist/mnist_test/mnist_test_label", 1, 45)
exp_v = "1.1.0"
output_dir = "infer_res/"+exp_v+'/'
os.makedirs(output_dir+'correct', exist_ok=True)
os.makedirs(output_dir+'incorrect', exist_ok=True)
iter = 60000
model_dir = "model/" + exp_v + "/" + str(iter) + '/'
res = []
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(model_dir+str(iter)+'.meta')
    new_saver.restore(sess, model_dir+str(iter))
    # y = tf.argmax(tf.get_collection('pred_network'), 1)
    y = tf.get_collection('pred_network')
    graph = tf.get_default_graph()

    x = graph.get_operation_by_name('input').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
    for i in range(data.size_test):
        res.append(sess.run([y], feed_dict={x: data.test_x[i].reshape(1, 45*45),  keep_prob: 1.0}))
    # res = y.eval(feed_dict={x: data.test_x[0:2], label: data.test_y[0:2],  keep_prob: 1.0}, session=sess)

infer_res = []
for i in range(data.size_test):
    infer_res.append(np.where(res[i] == np.max(res[i]))[3][0])
    # print(res[i])
    # print(np.where(res[i] == np.max(res[i]))[3][0])

# res = np.where(res == 1)[1]
count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(data.size_test):
    img = Image.fromarray(data.test_x[i].reshape(45, 45).astype(np.uint8))
    if infer_res[i] == data.test_y_no_one_hot[i]:
        img.save(output_dir+"correct/"+str(i)+".png")
    else:
        img.save(output_dir+"incorrect/"+str(i)+".png")
        count[data.test_y_no_one_hot[i]] += 1

print(count)
