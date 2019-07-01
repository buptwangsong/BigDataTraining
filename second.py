import numpy as np
import pandas as pd

d = np.random.random((1000,3))
print(d)
print(d.shape)
print(type(d))
data = pd.DataFrame(data=d, columns=list('家春秋'), index=np.arange(2019, 2019-d.shape[0], -1))
print(data)
data['result'] = data['家'] ** 2 - 3*data['家'] + 4*data['秋']
data.to_excel('巴金.xls')
