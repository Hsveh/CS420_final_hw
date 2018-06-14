"""
    data.py

"""
import common
import numpy as np
from sklearn.preprocessing import scale
from PIL import Image


def op0():
    """
    generate lowest 15000 confident images
    :return: none
    """
    data = common.Data("mnist/mnist_train/train_data.npy", "mnist/mnist_train/mnist_train_label",
                       "mnist/mnist_test/test_data.npy", "mnist/mnist_test/mnist_test_label", 1, 28)

    res = common.predict('model/1.4.0', 60000, data.train_x, 28)
    common.gen_data(res, data.train_x, data.train_y_no_one_hot, 15000)


def op1():
    """
    generate fc2 output for svm
    :return: none
    """
    data = common.Data("mnist/mnist_train/train_data.npy", "mnist/mnist_train/mnist_train_label",
                       "mnist/mnist_test/test_data.npy", "mnist/mnist_test/mnist_test_label", 1, 28)

    res = common.predict('CNN/model/SVM1', 60000, data.test_x, 28, "out1")

    data_fc = []
    for i in range(len(res)):
        data_fc.append(res[i][0][0][0])

    data_fc = np.array(data_fc)
    np.save('mnist/mnist_test/fc1_5.npy', data_fc)


def op2():
    """
    generate rotation data
    :return:
    """
    data = common.Data("mnist/mnist_train/train_data.npy", "mnist/mnist_train/mnist_train_label",
                       "mnist/mnist_test/test_data.npy", "mnist/mnist_test/mnist_test_label", 1, 28)
    fig = 28
    d = []
    l = []
    for i in range(len(data.train_x)):
        data.train_x[i] = data.train_x[i].astype(np.uint8)
        img = Image.fromarray(data.train_x[i].reshape(fig, fig).astype(np.uint8))
        img_r = img.rotate(45)
        img_l = img.rotate(315)
        matrix = np.asarray(img_r)
        matrix = matrix.reshape(fig*fig,)
        d.append(data.train_x[i])
        d.append(matrix)
        matrix = np.asarray(img_l)
        matrix = matrix.reshape(fig * fig, )
        d.append(matrix)
        l.append(data.train_y_no_one_hot[i])
        l.append(data.train_y_no_one_hot[i])
        l.append(data.train_y_no_one_hot[i])

    np.save('mnist/mnist_train/rotate_train.npy', d)
    np.save('mnist/mnist_train/rotate_label.npy', l)


def op3():
    """
    generate 14*14 for svm
    :return:
    """
    a = np.load('mnist/mnist_test/14.npy')
    res = []
    for i in range(len(a)):
        t = scale(a[i])
        t = t.reshape(28, 28)
        temp = np.zeros(shape=(14, 14))
        for j in range(14):
            for k in range(14):
                temp[j][k] = np.max(t[j*2:(j+1)*2, k*2:(k+1)*2])
        temp = temp.reshape(14*14,)
        res.append(temp)
    res = np.array(res)
    np.save('mnist/mnist_test/14.npy', res)
    for i in range(1):
        print(a[i])


if __name__ == '__main__':
    op0()
    op1()
    op2()
    op3()
