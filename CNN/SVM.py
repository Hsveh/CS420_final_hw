"""
Read data from CNN_SVM
"""
import common
import numpy as np
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

data = common.Data("../mnist/mnist_train/train_data.npy", "../mnist/mnist_train/mnist_train_label",
                   "../mnist/mnist_test/test_data.npy", "../mnist/mnist_test/mnist_test_label", 1, 28)
train = np.load('../mnist/mnist_train/fc1_5.npy')
test = np.load('../mnist/mnist_test/fc1_5.npy')

train = scale(train)
test = scale(test)

clf = SVC(kernel='rbf')
clf.fit(train, data.train_y_no_one_hot)
y_pred = clf.predict(test)

print(classification_report(data.test_y_no_one_hot, y_pred))
print(accuracy_score(data.test_y_no_one_hot, y_pred))
