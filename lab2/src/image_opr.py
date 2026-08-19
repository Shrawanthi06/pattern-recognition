from PIL import Image
import numpy as np

def load_img(filename):
	return Image.open(filename)

def resize_sqr_img(image,size=100):
	return image.resize((size,size))

def resize_rect_img(image,max_dimension=100):
	width,height=image.size
	scale=max_dimension/max(width,height)
	new_width=int(round(width*scale))
	new_height=int(round(height*scale))
	return image.resize((new_width,new_height))

def greyscale(image):
	return image.convert("L")

def img_to_matrix(image):
	return np.asarray(image,dtype=float)

def matrix_to_img(matrix):
	matrix=np.real_if_close(matrix) #remove tiny imaginary numerical eroors
	matrix=np.real(matrix) #take only real part in the complex begore img conversion
	matrix=np.clip(matrix,0,255) #greyscale range
	matrix=matrix.astype(np.uint8)
	return Image.fromarray(matrix,mode="L")

def save_matrix_as_img(matrix,filename):
	image=matrix_to_img(matrix)
	image.save(filename)

def preprocess_square(filename,size=100):
	image=load_img(filename)
	resized=resize_sqr_img(image,size)
	grayscaled=greyscale(resized)
	matrix=img_to_matrix(grayscaled)
	return matrix

def preprocess_rectangle(filename,max_dimension=100):
	image=load_img(filename)
	resized=resize_rect_img(image,max_dimension)
	grayscaled=greyscale(resized)
	matrix=img_to_matrix(grayscaled)
	return matrix

