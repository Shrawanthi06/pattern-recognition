import numpy as np

def frobenious_norm(A):
	return np.sqrt(np.sum(np.abs(A)**2))

def reconstruction_error(A,Ak):
	return frobenious_norm(A-Ak)

def relative_error(A,Ak):
	return(frobenious_norm(A-Ak)/frobenious_norm(A))

def error_img(A,Ak):
	return np.abs(A-Ak) #element wise error img

def energy_captured(singular_values,k):
	total_energy=np.sum(singular_values**2)
	retained_energy=np.sum(singular_values[:k]**2)
	return retained_energy/total_energy

def find_k_for_energy(singular_values, target):
	cumulative=np.cumsum(singular_values**2)
	total=np.sum(singular_values**2)
	fraction=cumulative/total
	k=np.argmax(fraction>=target)+1
	return int(k)

def compression_ratio(m,n,k):
	og=m*n
	compressed=((m*k)+k+(n*k))
	return og/compressed


