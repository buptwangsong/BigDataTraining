import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC


data = pd.read_csv('wine.data', header=None)
m, n = data.shape
x = data[np.arange(1,n)]
y = data[0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# model = LogisticRegression(penalty='l2', C=1.0)
# model = GridSearchCV(LogisticRegression(), cv=5, param_grid={
#     'penalty': ['l1', 'l2'],
#     'C': np.logspace(-4, 4, 20)
# })
model = SVC(kernel='linear', C=1.0)
model.fit(x_train, y_train)
# print('最优参数：', model.best_params_)
y_train_pred = model.predict(x_train)
# print('训练集正确率：', np.mean(y_train == y_train_pred))
print('训练集正确率：', accuracy_score(y_train, y_train_pred))
print('训练集混淆矩阵：\n', confusion_matrix(y_train, y_train_pred))
y_test_pred = model.predict(x_test)
# print('测试集正确率：', np.mean(y_test == y_test_pred))
print('测试集正确率：', accuracy_score(y_test, y_test_pred))
print('测试集混淆矩阵：\n', confusion_matrix(y_test, y_test_pred))
