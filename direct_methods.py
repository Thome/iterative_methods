import numpy as np

def LU(A):
	n = A.shape[1]
	L = np.eye(n)
	U = np.zeros((n,n))
	for k in range(n):
		for j in range(k,n):
			U[k,j] = A[i,j]
			for s in range(k):
				U[k,j] = U[k,j] - L[k,s]*U[s,j]
		for i in range(k+1,n):
			L[i,k] = A[i,k]
			for s in range(k):
				L[i,k] = L[i,k] - L[i,s]*U[s,k]
			L[i,k] = L[i,k]/U[k,k]
	return L, U

def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A
 
def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

def sor_solver(A, b, omega, initial_guess, convergence_criteria):
  """
  Entradas:
    A: matriz nxn
    b: vetor n dimensional
    omega: relaxation factor
    initial_guess: aproximação inicial
  Returns:
    phi: vetor solução n dimensional
  """
  phi = initial_guess[:]
  residual = np.linalg.norm(np.matmul(A, phi) - b) #Initial residual
  while residual > convergence_criteria:
    for i in range(A.shape[0]):
      sigma = 0
      for j in range(A.shape[1]):
        if j != i:
          sigma += A[i][j] * phi[j]
      phi[i] = (1 - omega) * phi[i] + (omega / A[i][i]) * (b[i] - sigma)
    residual = np.linalg.norm(np.matmul(A, phi) - b)
    print('Residual: {0:10.6g}'.format(residual))
  return phi

