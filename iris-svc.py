import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

model_name = 'iris.svm'
data = pd.read_csv('iris.data', header=None)
data[4] = LabelEncoder().fit_transform(data[4])
x = data[[0,1,2,3]]
y = data[4]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
if os.path.exists(model_name):
    print('加载模型...')
    model = joblib.load(model_name)
else:
    print('训练模型...')
    # model = RandomForestClassifier(n_estimators=20, max_depth=5, min_samples_split=3)
    model = SVC(C=1.0, kernel='linear')
    # model = GridSearchCV(LogisticRegression(), cv=3, param_grid={
    #     'penalty': ['l1', 'l2'],
    #     'C': np.logspace(0, 4, 10)
    # })
    model.fit(x_train, y_train)
    joblib.dump(model, model_name)
    # print('最优参数：', model.best_params_)
y_train_pred = model.predict(x_train)
# print('训练集正确率：', np.mean(y_train_pred == y_train))
print('训练集正确率：', accuracy_score(y_train, y_train_pred))
y_test_pred = model.predict(x_test)
# print('测试集正确率：', np.mean(y_test_pred == y_test))
print('测试集正确率：', accuracy_score(y_test, y_test_pred))
