import numpy as np
from math import sqrt
from scipy.linalg import svd

def jacobi(A,b,XO,TOL,N_iter):
    n = A.shape[1]
    k = 1
    x = np.array([0]*XO.size,dtype=float)
    while (k <= N_iter):
        for i in range(0,n):
            aux = [A[i,j] * XO[j] for j in range(0,n) if j != i]
            soma = -sum(aux) + b[i]
            x[i] = (1/A[i,i]) * soma

        v = x - XO
        norm = max(np.absolute(v))
        if norm < TOL:
            print("The procedure was successful")
            return x

        k = k+1

        for i in range(0,n):
            XO[i] = x[i]

    print("Maximum number of iterations exceeded")
    return 0

def gauss_seidel(A,b,XO,TOL,N_iter):
    n = A.shape[1]
    k = 1
    x = np.array([0]*XO.size,dtype=float)
    while (k <= N_iter):

        for i in range(0,n):
            aux1 = sum([A[i,j]*x[j] for j in range(0,i)])
            aux2 = sum([A[i,j]*XO[j] for j in range(i+1,n)])
            x[i] = (1/A[i,i]) * (-aux1 - aux2 + b[i])

        v = x - XO
        norm = max(np.absolute(v))
        if norm < TOL:
            print("The procedure was successful")
            return x

        k = k+1

        for i in range(0,n):
            XO[i] = x[i]

    print("Maximum number of iterations exceeded")
    return 0

def SQR(A,b,XO,w,TOL,N_iter):
    n = A.shape[1]
    k = 1
    x = np.array([0]*XO.size,dtype=float)
    while (k <= N_iter):

        for i in range(0,n):
            aux1 = -sum([A[i,j]*x[j] for j in range(0,i)])
            aux2 = -sum([A[i,j]*XO[j] for j in range(i+1,n)])
            aux = (1/A[i,i]) * (w*(aux1+aux2+b[i]))
            x[i] = (1-w)*XO[i] + aux

        v = x - XO
        norm = max(np.absolute(v))
        if norm < TOL:
            print("The procedure was successful")
            return x

        k = k+1

        for i in range(0,n):
            XO[i] = x[i]

    print("Maximum number of iterations exceeded")
    return 0

def is_edd(A):
    m,n = A.shape
    for i in range(m):
        diag = abs(A[i,i])
        soma = sum([abs(A[i,j]) for j in range(n) if j != i])
        if (not diag > soma):
            return 0
    return 1

def jacobi_spectral(A):
    m,n = A.shape
    D = np.ndarray(shape=(m,n),dtype=float)
    LU = np.ndarray(shape=(m,n),dtype=float)
    for i in range(m):
        for j in range(n):
            if (i==j):
                D[i,j] = A[i,j]
                LU[i,j] = 0
            else:
                D[i,j] = 0
                LU[i,j] = A[i,j]
    Dinv = np.linalg.inv(D)
    P = np.dot(-Dinv,LU)
    w,_ = np.linalg.eig(P)
    spectral = max(np.absolute(w))
    print("Spectral: ", spectral)

def gauss_spectral(A):
    m,n = A.shape
    DL = np.ndarray(shape=(m,n),dtype=float)
    U = np.ndarray(shape=(m,n),dtype=float)
    for i in range(m):
        for j in range(n):
            if (i>=j):
                DL[i,j] = A[i,j]
                U[i,j] = 0
            else:
                DL[i,j] = 0
                U[i,j] = A[i,j]
    DLinv = np.linalg.inv(DL)
    P = np.dot(-DLinv,U)
    w,_ = np.linalg.eig(P)
    spectral = max(np.absolute(w))
    print("Spectral: ", spectral)

def use_svd(A):
    #print(A)
    U, s, VT = svd(A)
    m,n = A.shape
    if (m!=n):
        Sigma = np.zeros(A.shape)
        Sigma[:A.shape[1], :A.shape[1]] = np.diag(s)
    else:
        Sigma = diag(s)
    B = U.dot(Sigma.dot(VT))
    return U, Sigma, VT
    #print(B)

def print_result(result):
    for i in range(len(result)):
        print(round(result[i],6))


A = np.array([[-1,0,0,sqrt(2)/2,1,0,0,0],
            [0,-1,0,sqrt(2)/2,0,0,0,0],
            [0,0,-1,0,0,0,0.5,0],
            [0,0,0,-sqrt(2)/2,0,-1,-0.5,0],
            [0,0,0,0,-1,0,0,1],
            [0,0,0,0,0,1,0,0],
            [0,0,0,-sqrt(2)/2,0,0,sqrt(3)/2,0],
            [0,0,0,0,0,0,-sqrt(3)/2,-1]])

b = np.array([0,0,0,0,0,10000,0,0])
XO = np.array([0,0,0,0,0,0,0,0],dtype=float)
TOL = 0.01
N_iter = 150

result = jacobi(A,b,XO,TOL,N_iter)
print_result(result) #Imprime o resultado de Jacobi
# Resultado: 
#-0.002651
#-6339.744637
#-3660.255363
#-8965.752847
#6339.748258
#10000.0
#-7320.506545
#6339.748258

XO = np.array([0,0,0,0,0,0,0,0],dtype=float)
result = gauss_seidel(A,b,XO,TOL,N_iter)
print_result(result) #Imprime o resultado de Gauss
# Resultado:
#0.003621
#-6339.744637
#-3660.253273
#-8965.755804
#6339.744637
#10000.0
#-7320.508959
#6339.746727

XO = np.array([0,0,0,0,0,0,0,0],dtype=float)
w = 1.25
result = SQR(A,b,XO,w,TOL,N_iter)
print(result) #Imrpime o resultado de SQR (w = 1.25)
# Resultado:
#Maximum number of iterations exceeded

XO = np.array([0,0,0,0,0,0,0,0],dtype=float)
w = 1.1
result = SQR(A,b,XO,w,TOL,N_iter)
print_result(result) #Imrpime o resultado de SQR (w = 1.1)
# Resultado:
#0.004513
#-6339.744528
#-3660.253012
#-8965.756173
#6339.743759
#10000.0
#-7320.509545
#6339.747539

T = np.array([[5,-1,3],
              [2,-8,1],
              [-2,0,4]])

T1 = np.array([[1., 0., 0.],
               [0., 2., 0.],
               [0., 0., 3.]])

jacobi_spectral(A) #Para achar o raio espectral segundo Jacobi
#Spectral:  0.7598356856515925 
gauss_spectral(A) #Para achar o raio espectral segundo Gauss
#Spectral:  0.5773502691896258

T2 = np.array([[1,0],[0,sqrt(2)],[0,sqrt(2)]])
U, Sigma, VT = use_svd(T2)
print(U)
#[[ 0.          1.          0.        ]
# [-0.70710678  0.         -0.70710678]
# [-0.70710678  0.          0.70710678]]
print(Sigma)
#[[2. 0.]
# [0. 1.]
# [0. 0.]]

print(VT)
#[[-0. -1.]
# [ 1.  0.]]
