def transpose(A):
    #transpose of design matrix A
    m=len(A)
    n=len(A[0])
    A_T=[[0.0 for _ in range(m)]for _ in range(n)]
    for i in range(m):
        for j in range(n):
            A_T[j][i]=A[i][j]
    return A_T

def matmul(A,B):
    #compute A^tA
    m=len(A)
    k_A=len(A[0])
    k_B=len(B)
    n=len(B[0])
    if k_A!=k_B:
        raise ValueError(f"Shape mismatch in matrix multiplication: A has shape ({m},{k_A}), B has shape ({k_B},{n}):({m}x{k_A}) and ({k_B}x{n})")
    C=[[0.0 for _ in range(n)]for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total=0.0
            for p in range(k_A):
                total += A[i][p]*B[p][j]
            C[i][j]=total
    return C

def matvec(A,v):
    #compute A^tA
    m=len(A)
    n=len(A[0])
    if len(v)!=n:
        raise ValueError(f"shape mismatch in matvec: matrix has {n} cols, vector has {len(v)} elements")
    result=[0.0 for _ in range(m)]
    for i in range(m):
        total=0.0
        for j in range(n):
            total += A[i][j]*v[j]
        result[i]=total
    return result

def identity(n):
    #create identity matrix
    I=[[0.0 for _ in range(n)]for _ in range(n)]
    for i in range(n):
        I[i][i]=1.0
    return I

def solve_linear_system(A,b):
    #solve Ax=b using Gaussian elimination with partial pivoting
    n=len(A)
    #create augmented matrix A|b of size nx(n+1)
    aug=[]
    for i in range(n):
        aug.append([float(val) for val in A[i]]+[float(b[i])])
    #gauss-jordan elimination
    for col in range(n):
        #find row with max abs value in this column
        max_row=col
        max_val=abs(aug[col][col])
        for row in range(col+1,n):
            if abs(aug[row][col])>max_val:
                max_val=abs(aug[row][col])
                max_row=row
        if max_val<1e-14:
            raise ValueError("matrix is singular or nearly singular, so cannot solve system")
        #swap current row with max_row
        if max_row != col:
            aug[col],aug[max_row]=aug[max_row],aug[col]
        #scale pivot row so aug==1.0
        pivot=aug[col][col]
        for j in range(col,n+1):
            aug[col][j]/=pivot
        #eliminate all other rows in this column
        for row in range(n):
            if row!=col:
                factor=aug[row][col]
                if abs(factor)>1e-15:
                    for j in range(col,n+1):
                        aug[row][j]-=factor*aug[col][j]
    #extract solution x from last column
    x=[aug[i][n] for i in range(n)]
    return x

def matrix_inverse(A):
    n=len(A)
    aug=[]
    for i in range(n):
        row=[float(val) for val in A[i]]+[1.0 if j==i else 0.0 for j in range(n)]
        aug.append(row)
    for col in range(n):
        max_row=col
        max_val=abs(aug[col][col])
        for row in range(col+1,n):
            if abs(aug[row][col])>max_val:
                max_val=abs(aug[row][col])
                max_row=row
        if max_val<1e-14:
            raise ValueError("matrix is singular and cannot be inverted")
        if max_row != col:
            aug[col],aug[max_row]=aug[max_row],aug[col]
        pivot=aug[col][col]
        for j in range(col,2*n):
            aug[col][j]/=pivot
        for row in range(n):
            if row!=col:
                factor=aug[row][col]
                if abs(factor)>1e-15:
                    for j in range(col,2*n):
                        aug[row][j]-=factor*aug[col][j]
    inv=[[aug[i][j+n] for j in range(n)]for i in range(n)]
    return inv

if __name__=="__main__":
    print("testing the block of code")
    A = [[2.0,1.0], [1.0,3.0]]
    b = [5.0,10.0]
    solution=solve_linear_system(A, b)
    print(f"System Solution [x, y]: {solution} (Expected:[1.0, 3.0])")

    A_inv = matrix_inverse(A)
    product = matmul(A, A_inv)
    print(f"A * A^(-1) is approximately Identity:")
    for row in product:
        print("  ", [round(val, 6) for val in row])
    print("All Linear Algebra tests passed successfully!")