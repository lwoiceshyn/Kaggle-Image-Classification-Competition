import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder



data = np.load('data_batch_train.npz')
test = np.load('data_batch_test.npz')
df = pd.read_csv('train.csv')
label = df['Label']
y_train = label.as_matrix()
X_train = data['representations']
X_test = test['representations']

'''
Uncomment to use PCA for faster computation times (useful for CV)
'''
# pca = PCA(n_components=128)
# pca.fit(X_train)
# X_train = pca.transform(X_train)

'''
Uncomment desired classifier type
'''
# clf = svm.SVC(kernel='linear', C=10)
# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#               decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
#                max_iter=-1, probability=False, random_state=None, shrinking=True,
#                tol=0.001, verbose=False)
# clf = svm.SVC(C=14, kernel='rbf', gamma=0.001, cache_size=200)
clf = svm.SVC(C=6.2, kernel='poly', degree=4, coef0=0.48, cache_size=200)
# clf =   linear_model.SGDClassifier(alpha=0.1, average=False, class_weight=None, epsilon=0.1,
#         eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#         learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
#         penalty='l2', power_t=0.5, random_state=None, shuffle=True,
#         verbose=0, warm_start=False)
# clf = AdaBoostClassifier(n_estimators=1000)


scores = cross_val_score(clf, X_train, y_train, cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)


'''
Comment or uncomment to save model, and change name of output as desired
'''
np.savetxt('polysvm.csv', predictions)

