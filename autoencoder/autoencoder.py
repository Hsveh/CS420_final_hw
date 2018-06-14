"""
Autoencoder with convolutional and transposed convolutional layers.
SVM connected to the output of the encoder
"""

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.optimizers import SGD
from keras.utils import to_categorical

import common


"""
Variable Definition

train_num: training set size
test_num: testing set size
fig_w: figure width
epochs: number of epoches
batch_size: batch size
common_path = common path of input data
"""


train_num = 60000
test_num = 10000
fig_w = 28
epochs = 50
batch_size = 128
common_path = "../mnist"


def Autoencoder(train_x, test_x, train_y, test_y):
	""" Autoencoder training with SVM connected
	:param path: data file
	:return: data(np.array)
	"""

	# data preparation
	train_x = train_x.reshape(train_num, fig_w, fig_w,1)
	test_x = test_x.reshape(test_num, fig_w, fig_w,1)

	inputs = Input(shape=(fig_w, fig_w, 1))

	x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same', name='encoded')(x)

	x = Conv2DTranspose(16, (3, 3), activation='relu')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

	model = Model(inputs, decoded)

	model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
	model.fit(train_x, train_x, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(test_x, test_x))

	print("autoencoder training over")

	# extract encoder
	encoded_model = Model(inputs=model.input, outputs=model.get_layer('encoded').output)

	train_x = train_x.reshape(train_num, fig_w, fig_w, 1)
	test_x = test_x.reshape(test_num, fig_w, fig_w, 1)

	train_output = encoded_model.predict(train_x)
	test_output = encoded_model.predict(test_x)

	train_output = train_output.reshape(train_num, 128)
	test_output = test_output.reshape(test_num, 128)

	# train SVM classifier
	clf=svm.SVC(kernel='rbf', decision_function_shape='ovr')
	clf.fit(train_output, train_y)
	pred_y = clf.predict(test_output)
	print(clf.score(test_output,test_y))

	prec = precision_score(test_y,pred_y,average='weighted')
	recall = recall_score(test_y,pred_y,average='weighted')
	f1 = f1_score(test_y,pred_y,average='weighted')

	print('precision:\t'+str(prec))
	print('recall:\t'+str(recall))
	print('f1 score:\t'+str(f1))


def main():
	data = common.Data(common_path+"/mnist_train/train_data.npy", common_path+"/mnist_train/mnist_train_label",
					common_path+"/mnist_test/test_data.npy", common_path+"/mnist_test/mnist_test_label", fig)
	train_x = data.train_x
	test_x = data.test_x
	train_y = data.train_y
	test_y = data.test_y
	Autoencoder(train_x, test_x, train_y, test_y)


main()