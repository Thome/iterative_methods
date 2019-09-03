import numpy as np
from math import sqrt

def LU(A):
	n = A.shape[1]
	L = np.eye(n)
	U = np.zeros((n,n))
	for k in range(n):
		for j in range(k,n):
			U[k,j] = A[k,j]
			for s in range(k):
				U[k,j] = U[k,j] - L[k,s]*U[s,j]
		for i in range(k+1,n):
			L[i,k] = A[i,k]
			for s in range(k):
				L[i,k] = L[i,k] - L[i,s]*U[s,k]
			L[i,k] = L[i,k]/U[k,k]
	return L, U

def cholesky(A):
  n = A.shape[1]
  H = np.tril(A)
  for k in range(n-1):
    H[k,k] = sqrt(H[k,k])
    H[k+1:n,k] = H[k+1:n,k]/H[k,k]
    for j in range(k+1,n):
      H[j:n,j] = H[j:n,j]-H[j:n,k]*H[j,k]
  H[n-1,n-1] = sqrt(H[n-1,n-1])
  return H

def qr(A):
    n = A.shape[1]
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    for k in range(n):
      u = A[:,k]
      for j in range(k):
        u = u - np.dot(A[:,k],Q[:,j])*Q[:,j]
      unorm = sqrt(np.sum(np.square(u)))
      Q[:,k] = u/unorm
      for j in range(k,n):
        R[k,j] = np.dot(A[:,j],Q[:,k])
    return Q,R

A_LU = np.array([[4,3],[6,3]],dtype=float)
L, U = LU(A_LU)

A_cho = np.array([[3,4,3],[4,8,6],[3,6,9]],dtype=float)
H = cholesky(A_cho)

A_qr = np.array([[12,-51,4],[6,167,-68],[-4,24,-41]],dtype=float)
Q, R = qr(A_qr)

print("\n========")
print("DECOMPOSIÇÃO LU\n")
print("MATRIZ A:")
print(A_LU)
print("\nMATRIZ L RESULTADO:")
print(L)
print("\nMATRIZ U RESULTADO:")
print(U)

print("\n========")
print("DECOMPOSIÇÃO CHOLESKY\n")
print("MATRIZ A:")
print(A_cho)
print("\nMATRIZ H RESULTADO:")
print(H)


print("\n========")
print("DECOMPOSIÇÃO QR\n")
print("MATRIZ A:")
print(A_qr)
print("\nMATRIZ Q RESULTADO:")
print(Q)
print("\nMATRIZ R RESULTADO:")
print(R)