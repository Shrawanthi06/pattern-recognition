import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure the lab2 directory is in sys.path so imports like `from src...` work from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.image_opr import (preprocess_square, preprocess_rectangle, save_matrix_as_img)
from src.matrix_opr import (reconstruction_error, relative_error, error_img, energy_captured, find_k_for_energy, compression_ratio)
from src.svd_opr import (compute_SVD, reconstruct_SVD)
from src.evd_opr import (compute_EVD, sort_eigenvalues, select_components, reconstruct_EVD, reconstruct_EVD_reduced, verify_EVD)

# Paths relative to this script's directory
Input_dir = os.path.join(SCRIPT_DIR, "input")
Output_dir = os.path.join(SCRIPT_DIR, "output")
os.makedirs(Output_dir, exist_ok=True)

Square_img = os.path.join(Input_dir, "cat_03.png")
Rectangle_img = os.path.join(Input_dir, "cat_03_rect.png")

def print_separator(title):
	print()
	print("=" * 65)
	print(title)
	print("=" * 65)

#square image

print_separator("Square Image")
A=preprocess_square(Square_img,size=100)
m,n=A.shape
print("Square image matrix", A.shape)
save_matrix_as_img(A,os.path.join(Output_dir,"square_org.png"))

#EVD

print_separator("EVD of Square Image")
eigenvalues,Q=compute_EVD(A)
eigenvalues,Q=sort_eigenvalues(eigenvalues,Q)
print("\nfirst 10 eigenvalues:")
for value in eigenvalues[:10]:
	print(value)
complete_EVD_error=verify_EVD(A,eigenvalues,Q)
print("\n complete reconstruction error of EVD:", complete_EVD_error)

#SVD

print_separator("SVD of Square Image")
U,S,Vt=compute_SVD(A)
print("shape of U:",U.shape)
print("Number of singular values:",len(S))
print("shape of V transpose:",Vt.shape)
print("\nfirst 10 singular values:")
print(S[:10])

#automatic choice of k
k90=find_k_for_energy(S,0.90)
k95=find_k_for_energy(S,0.95)
k99=find_k_for_energy(S,0.99)
print_separator("Energy based component selection")
print("k with 90% energy:",k90)
print("k with 95% energy:",k95)
print("k with 99% energy:",k99)
k_values=sorted(set([k90,k95,k99]))
#manually add higher k values to test sharper reconstructions
#square image k is capped at min(m,n) = 100
EXTRA_K_VALUES=[30,50,80]
k_values=sorted(set(k_values+EXTRA_K_VALUES))

#SVD reconsstruction

print_separator("SVD reconstruction")
svd_error=[]
for k in k_values:
	Ak=reconstruct_SVD(U,S,Vt,k)
	error=reconstruction_error(A,Ak)
	relative=relative_error(A,Ak)
	energy=energy_captured(S,k)
	ratio=compression_ratio(m,n,k)
	print()
	print("k=",k)
	print(f"ENERGY RETAINED:" f"{energy*100:.2f}%")
	print(f"FROBENIOUS ERROR:" f"{error:.4f}")
	print(f"RELATIVE ERROR:" f"{relative*100:.2f}%")
	print(f"COMPRESSION RATIO:" f"{ratio:.2f}:1")
	save_matrix_as_img(Ak,os.path.join(Output_dir,f"square_svd_{k}.png"))
	E=error_img(A,Ak)
	save_matrix_as_img(E,os.path.join(Output_dir,f"square_svd_error_{k}.png"))
	svd_error.append(error)

#EVD reconstruction

print_separator("EVD reconstruction")
evd_error=[]
for k in k_values:
	selected=select_components(eigenvalues,k)
	Ak=reconstruct_EVD(eigenvalues,Q,selected)
	error=reconstruction_error(A,Ak)
	print()
	print("requested k=",k)
	print("retained:",len(selected))
	print(f"FROBENIOUS ERROR: " f"{error:.4f}")
	save_matrix_as_img(Ak, os.path.join(Output_dir,f"square_evd_{k}.png"))
	E=error_img(A,Ak)
	save_matrix_as_img(E,os.path.join(Output_dir,f"square_evd_error_{k}.png"))
	evd_error.append(error)
	Ak_reduced=reconstruct_EVD_reduced(eigenvalues,Q,selected)
	reduced_error=reconstruction_error(A,Ak_reduced)
	match_diff=np.max(np.abs(Ak-Ak_reduced))
	print(f"REDUCED-DIMENSION FROBENIOUS ERROR: "F"{reduced_error:.4F}")
	print(f"max difference vs full size reconstruction:" f"{match_diff:.2e}")

#Singular value spectrum

print_separator("SINGULAR VALUE SPECTRUM")
plt.figure()
plt.plot(range(1, len(S) + 1),S)
plt.xlabel("Component k")
plt.ylabel("Singular value")
plt.title("Singular Value Spectrum - Square Image")
plt.grid()
plt.savefig(os.path.join(Output_dir,"square_singular_values.png"))
plt.close()

#Energy curve

all_energy = []
for k in range(1,len(S) + 1):
	all_energy.append(energy_captured(S,k))
plt.figure()
plt.plot(range(1, len(S) + 1),np.array(all_energy) * 100)
plt.xlabel("Number of retained components k")
plt.ylabel("Energy retained (%)")
plt.title("SVD Energy Curve - Square Image")
plt.grid()
plt.savefig(os.path.join(Output_dir,"square_energy_curve.png"))
plt.close()

#SVD error for all k

all_svd_errors = []
for k in range(1,min(m, n) + 1):
	Ak = reconstruct_SVD(U,S,Vt,k)
	error = reconstruction_error(A,Ak)
	all_svd_errors.append(error)
plt.figure()
plt.plot(range(1, len(all_svd_errors) + 1),all_svd_errors)
plt.xlabel("Number of retained components k")
plt.ylabel("Frobenious reconstruction error")
plt.title("SVD Reconstruction Error - Square Image")
plt.grid()
plt.savefig(os.path.join(Output_dir,"square_svd_error_curve.png"))
plt.close()

# EVD ERROR FOR ALL k

all_evd_errors = []
actual_evd_k = []
for k in range(1,n + 1):
	selected = select_components(eigenvalues,k)
	Ak = reconstruct_EVD(eigenvalues,Q,selected)
	error = reconstruction_error(A,Ak)
	all_evd_errors.append(error)
	actual_evd_k.append(len(selected))
plt.figure()
plt.plot(actual_evd_k,all_evd_errors)
plt.xlabel("Number of retained eigen-components")
plt.ylabel("Frobenious reconstruction error")
plt.title("EVD Reconstruction Error - Square Image")
plt.grid()
plt.savefig(os.path.join(Output_dir,"square_evd_error_curve.png"))
plt.close()

# EVD VS SVD

plt.figure()
plt.plot(range(1, len(all_svd_errors) + 1),all_svd_errors,label="SVD")
plt.plot(actual_evd_k,all_evd_errors,label="EVD")
plt.xlabel("Number of retained components")
plt.ylabel("Frobenious reconstruction error")
plt.title("EVD vs SVD - Square Image")
plt.legend()
plt.grid()
plt.savefig(os.path.join(Output_dir,"square_evd_vs_svd.png"))
plt.close()

#Rectangle image

print_separator("RECTANGULAR IMAGE")
R = preprocess_rectangle(Rectangle_img,max_dimension=100)
mr, nr = R.shape
print("Rectangular image matrix:",R.shape)
save_matrix_as_img(R,os.path.join(Output_dir,"rectangle_original.png"))

#SVD for rectangle image

print_separator("SVD OF RECTANGULAR IMAGE")
Ur,Sr,Vtr = compute_SVD(R)
print("U shape:",Ur.shape)
print("Number of singular values:",len(Sr))
print("V^T shape:",Vtr.shape)

#energy value of rectangle image

rk90 = find_k_for_energy(Sr,0.90)
rk95 = find_k_for_energy(Sr,0.95)
rk99 = find_k_for_energy(Sr,0.99)
print("\n90% energy k =",rk90)
print("95% energy k =",rk95)
print("99% energy k =",rk99)
RECT_K_VALUES = sorted(set([rk90,rk95,rk99]))
#manually add higher k values to test sharper reconstructions
#rectangle image k is capped at min(mr,nr) = 56
EXTRA_RECT_K_VALUES=[20,35,36]
RECT_K_VALUES = sorted(set(RECT_K_VALUES+EXTRA_RECT_K_VALUES))

#Rectangle image reconstruction

for k in RECT_K_VALUES:
	Rk = reconstruct_SVD(Ur,Sr,Vtr,k)
	error = reconstruction_error(R,Rk)
	energy = energy_captured(Sr,k)
	ratio = compression_ratio(mr,nr,k)
	print()
	print("Rectangle k =",k)
	print(f"Energy retained: " f"{energy * 100:.2f}%")
	print(f"Frobenious error: " f"{error:.4f}")
	print(f"Compression ratio: " f"{ratio:.2f}:1")
	save_matrix_as_img(Rk,os.path.join(Output_dir,f"rectangle_svd_k{k}.png"))
	E = error_img(R,Rk)
	save_matrix_as_img(E,os.path.join(Output_dir,f"rectangle_svd_error_k{k}.png"))

#Rectangle singular value  spectrum

plt.figure()
plt.plot(range(1, len(Sr) + 1),Sr)
plt.xlabel("Component k")
plt.ylabel("Singular value")
plt.title("Singular Value Spectrum - Rectangular Image")
plt.grid()
plt.savefig(os.path.join(Output_dir,"rectangle_singular_values.png"))
plt.close()

print_separator("Finished")
print("All results have been saved in:")
print(Output_dir)

