import numpy as np

a = np.arange(1, 10)
print(a)
a = a[-3:]
print(a)
size = 3
c = np.zeros((2, size, size)).astype(np.longlong)
print(c)
print(a[0:-2])