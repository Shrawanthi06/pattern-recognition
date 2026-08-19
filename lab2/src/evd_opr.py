import numpy as np

def compute_EVD(A):
    eigenvalues, Q=np.linalg.eig(A)
    return eigenvalues,Q

def sort_eigenvalues(eigenvalues, Q):
    n=len(eigenvalues)
    magnitudes=np.abs(eigenvalues)
    order=list(range(n))
    for i in range(n):
        largest=i
        for j in range(i+1,n):
            if magnitudes[order[j]]>magnitudes[order[largest]]:
                largest=j
        order[i],order[largest]=order[largest],order[i]
    order=np.array(order)
    eigenvalues_sorted=eigenvalues[order]
    Q_sorted=Q[:,order]
    return eigenvalues_sorted,Q_sorted

def find_conjugate_partner(eigenvalues, index, tolerance=1e-8):
    target=np.conjugate(eigenvalues[index])
    for j, value in enumerate(eigenvalues):
        if j==index:
            continue
        if np.isclose(value, target, atol=tolerance):
            return j
    return None

def select_components(eigenvalues, requested_k):
    selected=[]
    for i in range(len(eigenvalues)):
        if i in selected:
            continue
        if len(selected)>=requested_k:
            break
        selected.append(i)
        eigenvalue=eigenvalues[i]
        if abs(eigenvalue.imag)>1e-8:
            partner=find_conjugate_partner(eigenvalues,i)
            if partner is not None and partner not in selected:
                selected.append(partner)
    return selected

def reconstruct_EVD(eigenvalues, Q, selected):
    Q_inverse = np.linalg.inv(Q)
    n = len(eigenvalues)
    Lambda_k = np.zeros((n, n),dtype=complex)
    for index in selected:
        Lambda_k[index, index] = eigenvalues[index]
    #Complex reconstruction
    Ak_complex = Q @ Lambda_k @ Q_inverse
    imaginary_error=np.max(np.abs(Ak_complex.imag))
    if imaginary_error>1e-8:
        print(f"Warning: imaginary component="f"{imaginary_error:.4e}")
    Ak=Ak_complex.real
    return Ak

def reconstruct_EVD_reduced(eigenvalues,Q,selected):
    Q_inverse=np.linalg.inv(Q)
    Qk=Q[:,selected] #nxk
    Lambda_k=np.diag(eigenvalues[selected])#kxk
    Qk_inv=Q_inverse[selected,:] #kxn
    Ak_complex = Qk @ Lambda_k @ Qk_inv #k values nxn
    imaginary_error=np.max(np.abs(Ak_complex.imag))
    if imaginary_error>1e-8:
        print(f"Warning: imaginary components="f"{imaginary_error:.4e}")
    Ak=Ak_complex.real
    return Ak

def verify_EVD(A, eigenvalues, Q):
    Lambda=np.diag(eigenvalues)
    Q_inverse=np.linalg.inv(Q)
    reconstructed=(Q @ Lambda @ Q_inverse)
    error=np.linalg.norm(A-reconstructed.real)
    return error




