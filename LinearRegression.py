import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_excel('./data/巴金.xls')

data['家2'] = data['家']**2
data['春2'] = data['春']**2
data['秋2'] = data['秋']**2

x = data[['家2', '春2', '秋2', '家', '春', '秋']]
y = data['result']
model = LinearRegression()
model.fit(x, y)
print('系数：', model.coef_)
print('截距：', model.intercept_)
y_pred = model.predict(x)
mse = np.mean((y_pred - y) ** 2)
mae = np.mean(np.abs(y_pred - y))
print(mse, mae)
