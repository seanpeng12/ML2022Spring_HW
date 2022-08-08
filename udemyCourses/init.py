# numPy is a good alternative of matlab
# linear algebra 多維度
import numpy as np 
arr1d = np.array([4,5,6])
print(arr1d)

arr2d_0 = np.array([
    [3,4,5],
    [6,7,8]
])
arr2d_1 = np.array([[1,1,1],[2,2,2]])
arr2d_2 = np.array([(1,1,1),(2,2,2),(2,2,2),(2,2,2)])
print(arr2d_0 + arr2d_1)
print(arr1d.dtype)
print(arr1d.ndim) 

arr3d = np.array(
    [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
)
print(arr3d.ndim) 
print(arr3d.size) # elements count