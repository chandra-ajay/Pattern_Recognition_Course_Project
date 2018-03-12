import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import math

num_person = 40
num_image = 5

row, column = scipy.misc.imread('gallery/s1/1.pgm',-1).shape #row is 112 and column is 92

matrix_a = np.empty([row*column,1]) #Creating the matrix A (without mean centering)

for i in range(1, num_person + 1): #Converting all the images into array and constructing matrix A
	for j in range(1, num_image + 1):
		path = 'gallery/s' + str(i) + '/' + str(j) + '.pgm'
		img = scipy.misc.imread(path, -1)
		img = img.reshape(row*column,1)
		matrix_a = np.column_stack((matrix_a,img)) #Stacking all the images into a single matrix
		matrix_a = matrix_a.astype(int)

matrix_a = matrix_a[:,1:] #Removing the junk values formed due to creation of an empty array

matrix_mean = (np.sum(matrix_a,1).reshape(row*column,1))/(num_person*num_image)

matrix_b = matrix_a - matrix_mean #Creating the matrix B (mean centered)

matrix_b_trans = matrix_b.transpose()

# matrix_s_orig = np.matmul(matrix_b, matrix_b_trans)
matrix_s_dummy = np.matmul(matrix_b_trans, matrix_b) #Multipying transpose with normal as the calculation of EV would be easier in smaller matrix

eig_val, eig_vec_dummy = np.linalg.eig(matrix_s_dummy/(num_person*num_image))

# Eigenvalues are same in both the cases

eig_vec_orig = np.empty([row*column,1])
a = np.empty([1,1]) #Stores the norm values of eigenvectors
for i in range(0, num_person*num_image): #Constructing the eigen vector of original covariance matrix
	temp = np.matmul(matrix_b, eig_vec_dummy[:,i])
	norm = np.linalg.norm(temp)
	eig_vec_orig = np.column_stack((eig_vec_orig, temp))
	a = np.append(a, norm)

eig_vec_orig = eig_vec_orig[:,1:]
a = a[1:]

eig_val_sort = eig_val.argsort() #Special type of sorting that stores the indices (Refer documentation)
eig_val_sort = eig_val_sort[::-1]

eig_val = np.sort(eig_val)	#Sorts the eig_val array after obtaining the necessary indices using argsort function
eig_val = eig_val[::-1] 

eig_vec_sort = np.empty([row*column,1])
for i in range(0, num_person*num_image): #Replacing vectors such that they correspond to sorted eigenvalues
	eig_vec_sort = np.column_stack((eig_vec_sort, eig_vec_orig[:,eig_val_sort[i]] ))
	img = eig_vec_sort[:,i+1].reshape(row, column)
	path = 'eigenfaces/' + 'face_' + str(i+1) + '.pgm';
	scipy.misc.imsave(path, img)
	eig_vec_sort[:,i+1] = eig_vec_sort[:,i+1]/a[eig_val_sort[i]]

#Displaying top 5 eigenfaces
fig = plt.figure(num = 'Top 5 Eigenfaces') #Creates a new figure
num = 230 #Used to define the position of subplot
for i in range(0,5):
	num = num + 1
	ax = fig.add_subplot(num) #Adds a subplot to the figure with given position
	path = 'eigenfaces/' + 'face_' + str(i+1) + '.pgm'
	img = scipy.misc.imread(path)
	ax.imshow(img, cmap='gray') #Adds the image to subplot

plt.show()

#Finding the number of eigenvectors needed to capture 95% of total variance

sum_eig_val = 0
perc = np.empty([1,1])
a = True
fig = plt.figure(num = 'Eigenfaces vs Variance') #Create and rename the figure
for i in range(0,num_person*num_image):
	sum_eig_val = sum_eig_val + eig_val[i]
	perc = np.append(perc, (sum_eig_val*100)/sum(eig_val))
	if (perc[i+1] >=  95 and (a == True)):
		print (i+1), 'eigenvectors are required to capture 95 percent of total variance'
		a = False
	plt.plot(i+1,perc[i],'ro')

plt.xlabel('Number of Eigenfaces')
plt.ylabel('%'' of variance captured')
plt.show()

#Constructing the test image with Eigenfaces

for i in range(1,3):
	path = 'face_input_' + str(i) + '.pgm'
	img = scipy.misc.imread(path,-1)
	img = img.reshape(row*column,1)
	test_vec = img - matrix_mean
	weight = np.matmul(test_vec.transpose(), eig_vec_sort[:,1:])

	top_1 = weight[0,0]*eig_vec_sort[:,1]
	top_1 = top_1.reshape(row,column)
	path = 'output/' + 'face_' + str(i) + '_top_1.pgm';
	scipy.misc.imsave(path, top_1)

	top_15 = np.matmul(weight[0, 0:15], eig_vec_sort[:,1:16].transpose())
	top_15 = top_15.transpose()
	top_15 = top_15.reshape(row,column)
	path = 'output/' + 'face_' + str(i) + '_top_15.pgm';
	scipy.misc.imsave(path, top_15)

	top_200 = np.matmul(weight[0,:], eig_vec_sort[:,1:].transpose())
	top_200 = top_200.transpose()
	top_200 = top_200.reshape(row, column)
	path = 'output/' + 'face_' + str(i) + '_top_200.pgm';
	scipy.misc.imsave(path, top_200)

for i in range(0,2):
	fig = plt.figure(num = 'Input face with Eigenfaces - Image ' + str(i+1)) #Creates a new figure
	num = 131 #Used to define the position of subplot

	ax = fig.add_subplot(num) #Adds a subplot to the figure with given position
	path = 'output/' + 'face_' + str(i+1) + '_top_1.pgm'
	img = scipy.misc.imread(path)
	ax.imshow(img, cmap='gray') #Adds the image to subplot

	num = num + 1
	ax = fig.add_subplot(num)
	path = 'output/' + 'face_' + str(i+1) + '_top_15.pgm'
	img = scipy.misc.imread(path)
	ax.imshow(img, cmap='gray') #Adds the image to subplot

	num = num + 1
	ax = fig.add_subplot(num)
	path = 'output/' + 'face_' + str(i+1) + '_top_200.pgm'
	img = scipy.misc.imread(path)
	ax.imshow(img, cmap='gray') #Adds the image to subplot

plt.show()
