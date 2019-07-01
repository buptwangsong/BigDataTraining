import numpy as np
print(np.sqrt(6*np.sum(1/np.arange(1, 1000))))

d = np.random.random((400, 300, 3))
d *= 255
d = d.astype(np.uint8)
print(d)
