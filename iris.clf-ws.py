import pandas as pd
from sklearn.preprocessing import  LabelEncoder
data = pd.read_csv('./data/iris.data', header=None)
print(data)

x = data[[0,1,2,3]]
y = data[4]
print(x)
print(y)

le = LabelEncoder()
le.fit(data[4])
print(le.classes_)
data[4] = le.transform(data[4])
print(data)
