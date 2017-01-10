import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA


labels = pd.read_excel('train_bonus.xls', header=None)
data = np.load('data_batch_train.npz')
y_train = labels.as_matrix()
y_train_multi = MultiLabelBinarizer().fit_transform(y_train)

y_train = y_train_multi[:5600,:]
y_test = y_train_multi[5600:,:]

pca = PCA(n_components=128)
X_train = data['representations']
pca.fit(X_train)
X_train = pca.transform(X_train)
x_train = X_train[:5600,:]
x_test = X_train[5600:7000,:]
x_test_predicted = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train, y_train).predict(x_test)

n = float(y_test.shape[0]*y_test.shape[1])
diff = abs(x_test_predicted - y_test)
sum = np.sum(diff)
acc = float((n-sum)/n)
print 'Accuracy:' + str(acc)


