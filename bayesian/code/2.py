import numpy as np
import matplotlib as mp
import scipy.misc
import Tkinter
import tkFileDialog
import math

c = 3 #Number of classes
train_images = 200
test_images = 100

Tkinter.Tk().withdraw() #Closes the root window

matrices_list = list() #To store the training images into 3 different matrices 
for i in range(0,c):
	folder = tkFileDialog.askdirectory()
	print "Processing:",folder
	a = np.empty([32*32,1]) 
	for j in range(1,train_images+1):
		file_path = str(folder) + '/' + str(j) + '.jpg'
		img = scipy.misc.imread(file_path)
		img = scipy.misc.imresize(img,0.25) #Reduces the image size to 25% of original
	 	img = img.reshape(32*32,1)
		a = np.hstack((a,img))
	a = a[:,1:]
	matrices_list.append(a)
	
mv_list = list()
cv = np.zeros([32*32, 32*32])	
for matrix in matrices_list:
	mean_vector = (np.sum(matrix,1))/float(200)
	mv_list.append(mean_vector)
	for i in range(0,train_images):
		x = matrix[:,i]
		x = x - mean_vector
		x = x.reshape(32*32,1)
		x_trans = x.reshape(1,32*32)
		cv = cv + np.matmul(x,x_trans)

cv = (cv/200.0) + 0.5*(np.identity(32*32)) #Same covariance matrix for all the classes

def disc(x_test):
	x_test_trans = x_test.transpose()
	tot_val = list()
	for i in range(0,c):  #Discriminant function for the same covariance matrix for all classes
		mean_vector_trans = mv_list[i].transpose()
		cv_inv = np.linalg.inv(cv)
		
		val_1 = np.matmul(cv_inv, mv_list[i])
		val_1 = np.matmul(val_1, x_test)
		
		val_2 = np.matmul(mean_vector_trans, cv_inv)
		val_2 = np.matmul(val_2, mv_list[i])
		val_2 = -0.5*val_2
		
		val = val_1 + val_2
		tot_val.append(val)
	return tot_val.index(max(tot_val)) + 1 
		
for i in range(0,c):
	folder = tkFileDialog.askdirectory()
	print "Processing:",folder
	mis_cls = 0
	for j in range(1, test_images+1):
		file_path = str(folder) + '/' + str(200+j) + '.jpg'
		img = scipy.misc.imread(file_path)
		img = scipy.misc.imresize(img,0.25) #Reduces the image size to 25% of original
	 	img = img.reshape(32*32,1)
		class_label = disc(img)
		if class_label != (i+1):
			mis_cls = mis_cls + 1
	print "Number of misclassified images of class", (i+1), ":", mis_cls
	print "Accuracy:", ((test_images-mis_cls)/float(test_images))*100
