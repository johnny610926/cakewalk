import numpy as np
f = np.ndarray([])
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8,9],[10,11,12]])
c = np.concatenate((a, b)) # append(a,b,axis=0) is duplicated and actually it call concatenate
d = np.concatenate((c, a)) # append(c,a,axis=0) is duplicated
print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
print(d)

e = np.ndarray([])
print(e.ndim)