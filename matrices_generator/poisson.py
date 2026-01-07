import numpy as np
from scipy.sparse import diags, kron, eye
import scipy.io as sio



n = 2048
T = diags([ -1, 4, -1 ], offsets=[-1,0,1], shape=(n,n))
I = eye(n)
A = kron(I, T) + kron(diags([-1, -1], [-1,1], shape=(n,n)), I)
filename = f"poisson2d_n{n}.mtx"
sio.mmwrite(filename, A.tocsr())
print(f"生成 {filename}: {A.shape[0]}×{A.shape[1]} 矩阵，{A.nnz} 个非零元素")