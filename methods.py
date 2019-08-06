# A = [[a11, a12],[a21, a22]] mxn
# A[row] = [a21, a22]
# x = [x1, x2]
# b = [b1, b2]
# XO = [xo1, xo2] 
def jacobi(n,A,b,XO,TOL,N_iter):
    k = 1
    x = []
    while (k <= N_iter):

    	for i in range(0,n):
    		aux = [A[i][j] * XO[j] + b[i] for j in range(0,n) if j != i]
    		soma = -sum(aux)
    		x[i] = (1/A[i][i]) * soma

    	if abs(x - XO) < TOL:
    		print("The procedure was successful")
    		return x

    	k = k+1

    	for i in range(0,n):
    		XO[i] = x[i]

    print("Maximum number of iterations exceeded")
    return 0



n = 2

A = [[2,1],
     [5,7]]

b = [11,13]

XO = [1,1]

TOL = []

N_iter = 25

print(jacobi(n,A,b,XO,TOL,N_iter))

def gauss_seidel(n,A,b,XO,TOL,N_iter):
    k = 1
    x = []
    while (k <= N_iter):

        for i in range(0,n):
            aux1 = -sum([A[i][j]*x[j] for j in range(0,i-1)])
            aux2 = -sum([A[i][j]*XO[j] + b[i] for j in range(i+1,n)])
            x[i] = (1/A[i][i]) * (aux1+aux2)

        if abs(x - XO) < TOL:
            print("The procedure was successful")
            return x

        k = k+1

        for i in range(0,n):
            XO[i] = x[i]

    print("Maximum number of iterations exceeded")
    return 0

def SQR(n,A,b,XO,w,TOL,N_iter):
    k = 1
    x = []
    while (k <= N_iter):

        for i in range(0,n):
            aux1 = -sum([A[i][j]*x[j] for j in range(0,i-1)])
            aux2 = -sum([A[i][j]*XO[j] + b[i] for j in range(i+1,n)])
            aux = (1/A[i][i]) * (w*(aux1+aux2))
            x[i] = (1-w)*XO[i] + aux

        if abs(x - XO) < TOL:
            print("The procedure was successful")
            return x

        k = k+1

        for i in range(0,n):
            XO[i] = x[i]

    print("Maximum number of iterations exceeded")
    return 0