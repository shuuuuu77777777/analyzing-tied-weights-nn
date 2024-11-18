from scipy.linalg import pinv 
from scipy.linalg import inv 


import numpy as np

vector = np.arange(9)
vector = np.reshape(vector,[-1,1])
vector = vector[:3]
print(vector)


# array = np.arange(9)
# array = np.reshape(array,[3,3])
# print(array)

array = [[2,2,2],[1,1,1],[1,0,0]]
print(array)

array_pinv = pinv(array)
print(array_pinv)

print( np.linalg.matrix_rank(array_pinv))

array_PP = inv(array_pinv)


# print(array_pinv.dtype)

# print('array_pinv@vector')
# print(array_pinv@vector)

# print('array_pinv@array@vector')
# print(array_pinv@array@vector)

# print('array@array_pinv@vector')
# print(array@array_pinv@vector)

# print('array_pinv@array@vector + array@array_pinv@vector')
# print(array_pinv@array@vector + array@array_pinv@vector )

print(array_pinv @ array_PP )