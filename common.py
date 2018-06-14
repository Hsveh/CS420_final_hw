"""
    Author: Fanyong Xue, Dexin Yang, Shimian Zhang
    common.py
    contains some function that will be used on classifiers
    Data: a class to read data
    predict: infer data from model
    compare_result: compare inference result and label
    draw_image: draw debug image
    gen_data: generate data from inference result
    create_logger: create logger
    print_f: print function
"""

import tensorflow as tf
import numpy as np
import logging
import os
import sys
from PIL import Image


class Data():
    def __init__(self, train_x, train_y, test_x, test_y, batch_size, fig_w, logger=None):
        """data initiation
        :param train_x: path of train data
        :param train_y: path of train label
        :param test_x: path of test data
        :param test_y: path of test label
        :param batch_size: batch size
        :param fig_w: image size
        :param logger: logger
        """
        self.train_x_path = train_x
        self.train_y_path = train_y
        self.test_x_path = test_x
        self.test_y_path = test_y
        if (not os.path.exists(train_x)) or (not os.path.exists(train_y)) \
                or (not os.path.exists(test_x)) or (not os.path.exists(test_y)):
            if logger is None:
                print("Path Error!")
            else:
                logger.error("Path Error!")
            return
        self.batch_size = batch_size
        self.fig_w = fig_w
        self.ptr = 0
        self.size = 0
        self.size_test = 0
        self.train_x, self.train_y, self.test_x, self.test_y, self.train_y_no_one_hot, self.test_y_no_one_hot = self.read_data()
        if self.batch_size > self.size:
            return -1

    @staticmethod
    def read_file(path):
        """read file
        :param path: data file
        :return: data(np.array)
        """
        if ".npy" in path:
            data = np.load(path)
        else:
            data = np.fromfile(path, np.uint8)
        return data

    def read_data(self):
        """
        :return: train data(?, fig*fig), train label(one hot), test data(?, fig*fig)
                test label(one hot), train label(not one hot), test label(not one hot)
        """
        train_x = self.read_file(self.train_x_path)
        train_y = self.read_file(self.train_y_path)
        self.size = len(train_y)
        train_x = train_x.reshape(self.size, self.fig_w**2)
        test_x = self.read_file(self.test_x_path)
        test_y = self.read_file(self.test_y_path)
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
        return train_x, train_y_, test_x, test_y_, train_y, test_y

    def next_batch(self):
        """get next batch of train data and train label
        :return: next batch of train data and train label(one hot)
        """
        if self.ptr + self.batch_size >= self.size:
            head = 0
            tail = self.batch_size
            self.ptr = self.batch_size
        else:
            head = self.ptr
            tail = self.ptr + self.batch_size
            self.ptr += self.batch_size
        return self.train_x[head:tail, 0:self.fig_w**2], self.train_y[head:tail, 0:10]


def predict(model_dir, iteration, train, fig, variable="pred_network", logger=None):
    """predict by loading tensorflow model
    :param model_dir: model path, will load model_dir/iteration.meta and model_dir/iteration
    :param iteration: which iteration you want to use
    :param train: train data(shape=size*(fig**2))
    :param fig: image size
    :param variable: variable you want to collect
    :param logger: logger
    :return: inference result
    """
    meta = model_dir + '/' + str(iteration) + '/' + str(iteration) + '.meta'
    if not os.path.exists(meta):
        if logger is None:
            print('Error! '+meta+' not found')
        else:
            logger.error(meta+' not found')
        return
    restore = model_dir + '/' + str(iteration) + '/' + str(iteration)
    if not os.path.exists(restore):
        if logger is None:
            print('Error! '+restore+' not found')
        else:
            logger.error(restore+' not found')
        return

    res = []
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_dir + '/' + str(iteration) + '/' + str(iteration) + '.meta')
        new_saver.restore(sess, model_dir + '/' + str(iteration) + '/' + str(iteration))
        y = tf.get_collection(variable)
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        size = len(train)
        for i in range(size):
            res.append(sess.run([y], feed_dict={x: train[i].reshape(1, fig * fig), keep_prob: 1.0}))
    return res


def compare_result(res, label):
    """ compare result from inference and label
    :param res: inference result
    :param label: label data
    :return: inference result, inference score, label score
    """
    infer_res = []
    infer_conf = []
    gt_conf = []
    size = len(label)
    for i in range(size):
        infer_res.append(np.where(res[i] == np.max(res[i]))[3][0])
        infer_conf.append(res[i][0][0][0][infer_res[i]])
        gt_conf.append(res[i][0][0][0][label[i]])
    return infer_res, infer_conf, gt_conf


def draw_image(res, train, label, output_dir, fig, write_file=False):
    """draw debug image
    draw image by comparision reference and label, save image to output_dir/correct and output_dir/incorrect,
    if write_file is true, write scores to output_dir/res_correct.txt and output_dir/res_incorrect.txt
    :param res: reference result from model
    :param train: train data
    :param label: ground truth(not on hot)
    :param output_dir: where the image save
    :param fig: size of image
    :param write_file: weather to write confidence to file output_dir+/res_correct.txt or res_error.txt
    :return: none
    """
    size = len(label)
    infer_res, infer_conf, gt_conf = compare_result(res, label)
    if not os.path.exists(output_dir+"/correct"):
        os.makedirs(output_dir+"/correct")
    if not os.path.exists(output_dir + "/incorrect"):
        os.makedirs(output_dir + "/incorrect")

    if write_file:
        cor = open(output_dir+"/res_correct.txt", 'a')
        err = open(output_dir+"/res_incorrect.txt", 'a')
    # right inferences
    r = 0
    # error inferences
    e = 0
    # error details
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(size):
        img = Image.fromarray(train[i].reshape(fig, fig).astype(np.uint8))
        if infer_res[i] == label[i]:
            r += 1
            img.save(output_dir+"/correct/"+str(i)+"_"+str(infer_res[i])+"_"+str(infer_conf[i])+".png")
            if write_file:
                cor.write(str(i)+"_"+str(infer_res[i])+"_"+str(infer_conf[i])+str(res[i][0][0][0])+'\n')

        else:

            e += 1
            img.save(output_dir + "/incorrect/" + str(i) + "_gt:" + str(label[i]) + "_" + str(
                gt_conf[i]) + "_inf:" + str(infer_res[i]) + "_" + str(infer_conf[i]) + ".png")
            count[label[i]] += 1
            if write_file:
                err.write(str(i) + "_gt:"+str(label[i]) + "_"+str(gt_conf[i]) + "_inf:"+str(infer_res[i])+"_" +
                          str(infer_conf[i]) + str(res[i][0][0][0]) + '\n')
    if write_file:
        cor.close()
        err.close()
    print("correct inference results:"+str(r))
    print("incorrect inference results:" + str(e))
    print("incorrect inference results details:" + str(count))


def gen_data(res, train, label_no, num, save_path='../mnist/mnist_train'):
    """generate data
    generate the lowest num scores image
    :param res: inference result from predict
    :param train: predict data
    :param label_no: label(not on hot)
    :param num: number of final data
    :param save_path: common path of output data
    :return: none
    """
    size = len(label_no)
    infer_res, infer_conf, gt_conf = compare_result(res, label_no)
    threshold = sorted(infer_conf)[num]
    print("threshold:"+str(threshold))
    data_ = []
    label_ = []
    for i in range(size):
        if infer_conf[i] < threshold:
            data_.append(train[i].astype(np.uint8))
            label_.append(label_no[i])
    data_ = np.array(data_)
    label_ = np.array(label_)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path+'/data_'+str(num)+'.npy', data_)
    np.save(save_path+'/label_'+str(num)+'.npy', label_)


def create_logger(logger_name,
                  log_format=None,
                  log_level=logging.INFO,
                  log_path=None):
    """create a logger
    logger that prints or write file by specified format
    :param logger_name: logger name
    :param log_format: the format that you want the logger to use
    :param log_level: INFO,  ERROR or WARNING
    :param log_path: Log File, if none, logger will not write file
    :return:none
    """
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


def print_f(msg, i, t):
    """ print function
    show a progress bar in the cmd line
    Not available in IDE!!!!!!!
    :param msg: current process
    :param i: current iteration
    :param t: total iteration
    :return: none
    """
    if i == t-1:
        sys.stdout.write('\n')
        return
    t += 1
    i += 1
    msg += '\t'
    sys.stdout.write('\r')
    sys.stdout.write("%s%s%% |%s" % (msg, int(i % t), int(i % t) * '#'))
    sys.stdout.flush()
