"""
    Deep Residual Network
    Residual blocks
        32 layers: n=5, 56 layers: n=9, 110 layers: n=18
"""

from __future__ import division, print_function, absolute_import
import tflearn
import common

"""
Variable Definition
batch_size: batch size
fig: image size
max_epoch: max iteration
common_path = common path of input data
"""

n = 5
batch_size = 500
fig = 45
max_epoch = 200
common_path = "../mnist"
data = common.Data(common_path+"/mnist_train/mnist_train_data", common_path+"/mnist_train/mnist_train_label",
                   common_path+"/mnist_test/mnist_test_data", common_path+"/mnist_test/mnist_test_label", 1, fig)

X = data.train_x.reshape(data.size, fig, fig, 1)
Y = data.train_y
testX = data.test_x.reshape(data.size_test, fig, fig, 1)
testY = data.test_y

img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

img_aug = tflearn.ImageAugmentation()


net = tflearn.input_data(shape=[None, 45, 45, 1],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

net = tflearn.fully_connected(net, 10, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')

model = tflearn.DNN(net, checkpoint_path='model/',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=max_epoch, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=batch_size, shuffle=True,
          run_id='mnist')

