import tensorflow as tf
import os
import numpy as np
from PIL import Image
import sys
sys.path.append("../")
import common
fig = 28
data = common.Data("../mnist/mnist_train/train_data.npy", "../mnist/mnist_train/mnist_train_label", "../mnist/mnist_test/test_data.npy", "../mnist/mnist_test/mnist_test_label", 1, fig)
exp_v = "1.4.0"
output_dir = "infer_res/"+exp_v+'/'
# os.makedirs(output_dir+'correct', exist_ok=True)
# os.makedirs(output_dir+'incorrect', exist_ok=True)
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
    for i in range(data.size):
        res.append(sess.run([y], feed_dict={x: data.train_x[i].reshape(1, fig*fig),  keep_prob: 1.0}))
    # res = y.eval(feed_dict={x: data.test_x[0:2], label: data.test_y[0:2],  keep_prob: 1.0}, session=sess)

infer_res = []
infer_conf = []
gt_conf = []

for i in range(data.size):
    infer_res.append(np.where(res[i] == np.max(res[i]))[3][0])
    infer_conf.append(res[i][0][0][0][infer_res[i]])
    gt_conf.append(res[i][0][0][0][data.train_y_no_one_hot[i]])
    # print(res[i])
    # print(np.where(res[i] == np.max(res[i]))[3][0])

# res = np.where(res == 1)[1]
threshold = sorted(infer_conf)[5000]
print(threshold)
data_ = []
label = []
for i in range(data.size):
    if infer_conf[i] < threshold:
        data_.append(data.train_x[i].astype(np.uint8))
        label.append(data.train_y[i])
data_ = np.array(data_)
label = np.array(label)
np.save('data_10000.npy', data_)
np.save('label_10000.npy', label)
count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# cor = open(output_dir+"res_correct.txt", 'a')
# err = open(output_dir+"res_error.txt", 'a')
r = 0
e = 0
for i in range(data.size):
    #
    if infer_res[i] == data.train_y_no_one_hot[i]:
        # img.save(output_dir+"correct/"+str(i)+"_"+str(infer_res[i])+"_"+str(infer_conf[i])+".png")
        # cor.write(str(i)+"_"+str(infer_res[i])+"_"+str(infer_conf[i])+str(res[i][0][0][0])+'\n')
        r += 1
    else:
        img = Image.fromarray(data.train_x[i].reshape(fig, fig).astype(np.uint8))
        e += 1
        img.save(output_dir+"train_incorrect/"+str(i)+"_gt:"+str(data.train_y_no_one_hot[i])+"_"+str(gt_conf[i])+"_inf:"+str(infer_res[i])+"_"+str(infer_conf[i])+".png")
        #count[data.test_y_no_one_hot[i]] += 1
        #err.write(str(i)+"_gt:"+str(data.test_y_no_one_hot[i])+"_"+str(gt_conf[i])+"_inf:"+str(infer_res[i])+"_"+str(infer_conf[i])+str(res[i][0][0][0])+'\n')
# print(count)
# cor.close()
# err.close()
print(r)
print(e)
