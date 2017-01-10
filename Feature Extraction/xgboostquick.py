import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import xgboost
from sklearn.decomposition import PCA

data = np.load('data_batch_train.npz')
test = np.load('data_batch_test.npz')
df = pd.read_csv('train.csv')
label = df['Label']
y_train = label.as_matrix()
X_train = data['representations']
X_test = test['representations']

'''
Uncomment to use PCA for faster computation times (useful for testing parameters)
'''
# pca = PCA(n_components=128)
# pca.fit(X_train)
# X_train = pca.transform(X_train)

model = xgboost.XGBClassifier(learning_rate =0.05,
 n_estimators=3000,
 max_depth=4,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

scores = cross_val_score(model, X_train, y_train, cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
model.fit(X_train,y_train)
predictions = model.predict(X_test)

np.savetxt('xgboost.csv', predictions)

