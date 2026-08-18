import numpy as np

def compute_SVD(A):
	U,S,Vt=np.linalg.svd(A,full_matrices=False)
	return U,S,Vt

def reconstruct_SVD(U,S,Vt,k):
	Uk=U[:,:k]
	Sk=S[:k]
	Vtk=Vt[:k,:]
	Sigma_k=np.diag(Sk)
	Ak=(Uk @ Sigma_k @ Vtk)
	return Ak
