import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA  
from sklearn.decomposition import FastICA  
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import svm


def loadDataSet(train_data_path, train_label_path, test_data_path, test_label_path, train_data_num = 60000, test_data_num = 10000, fig_w = 45):
	train_data = np.load(train_data_path)
	train_label = np.fromfile(train_label_path,dtype=np.uint8)
	train_data = train_data.reshape(train_data_num,fig_w*fig_w)

	test_data = np.load(test_data_path)
	test_label = np.fromfile(test_label_path,dtype=np.uint8)
	test_data = test_data.reshape(test_data_num,fig_w*fig_w)

	return train_data,train_label,test_data,test_label

def pca_algorithm(dataSet , d = 30):
	print('Doing PCA......')
	print('Origin data size:'+str(dataSet.shape))
	pca = PCA(n_components=d)
	pca.fit(dataSet)
	new_data = pca.transform(dataSet)
	print('After PCA data size:'+str(new_data.shape))
	return new_data

def ica_algorithm(dataSet , d = 0.95):
	print('Doing FastICA......')
	print('Origin data size:'+str(dataSet.shape))
	ica = FastICA(n_components=d)
	ica.fit(dataSet)
	new_data = ica.transform(dataSet)
	print('After ICA data size:'+str(new_data.shape))
	return new_data

def svm_algorithm(train_data, train_label, test_data, test_label, pca = False):
	print('-----------SVM Algorithm-------------')
	
	if (pca == True):
		train_data = pca_algorithm(train_data)
		test_data = pca_algorithm(test_data,train_data.shape[1])
	clf = svm.SVC(decision_function_shape = 'ovr')
	print(clf)
	clf.fit(train_data, train_label)

	predict_label = clf.predict(test_data)

	print('Result:')
	print(classification_report(test_label, predict_label))

#train_data,train_label,test_data,test_label = loadDataSet('mnist_train_data', 'mnist_train_label', 'mnist_test_data', 'mnist_test_label')
train_data,train_label,test_data,test_label = loadDataSet('train_data.npy', 'mnist_train_label', 'test_data.npy', 'mnist_test_label', fig_w = 28)
print(train_data.shape)
print(train_label.shape)
svm_algorithm(train_data,train_label,test_data,test_label, pca=True)