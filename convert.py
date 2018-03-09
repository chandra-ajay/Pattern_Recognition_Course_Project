import scipy.misc
import matplotlib.pyplot as plt

for i in range(1,3):
	path = 'face_input_' + str(i) + '.pgm'
	img = scipy.misc.imread(path,-1)
	path = 'output/' + 'face_convert' + str(i) + '.pgm';
	scipy.misc.imsave(path, img)
	fig = plt.figure(num = 'Convert' + str(i)) #Creates a new figure
	num = 111 #Used to define the position of subplot
	
	ax = fig.add_subplot(num) #Adds a subplot to the figure with given position
	path = 'output/' + 'face_convert' + str(i) + '.pgm'
	img = scipy.misc.imread(path)
	ax.imshow(img, cmap='gray') #Adds the image to subplot
	
plt.show()