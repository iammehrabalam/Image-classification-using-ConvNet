
import numpy as np
from PIL import Image
import pickle 
def getTrainingDataset():
	f=open("../train_images/release/train_annos.txt","r")
	resizeImg(f)
	
   	#return list of tuples
   	#tuple contain numpyarray of image pixels, shoes class 	
	

'''     
pytable and HDF5

		op = open("train_images/shoesImg_nparray.pkl","rb")
        listOftuple = pickle.load(op)
        print len(listOftuple)
        op.close
'''
def resizeImg(f):
	s=[]
	for i in f:
    	#45542,train_images/0045/FGCOMP_0045542.jpg,5,500,143,501,55,417
		format_list = i.split(",")
		x = int(format_list[0])
		if x >=45542:
			try:
				filename = "../"+str(format_list[1])
				shoe_image = Image.open(filename)				
				shoe_image=shoe_image.resize((64,64),Image.ANTIALIAS)
				print x
				shoe_image=np.asarray(shoe_image,dtype="f")
				s.append((shoe_image,format_list[3]))
			
			except:
				print format_list[1] + '...skipped'	
	pkl=open("../train_images/shoesImg_nparray.pickle","wb")
	pickle.dump(s,pkl,pickle.HIGHEST_PROTOCOL)#for binary file protocol>=1
	pkl.close()


#count no of images in each class	
def countClasses(f):
	classCounter = [0]*70
	print classCounter

	for i in f:
		format_list=i.split(",")
		x=int(format_list[0])
		if x>=45542:
			#print format_list[1],format_list[3]			
			classCounter[int(format_list[3])-500]+=1
	print	classCounter
	top20 = sorted(enumerate(classCounter),key=lambda x:x[1])[-20:]
	print top20
	print [sum(x) for x in zip(*top20)]#sum of top20 images dataset


getTrainingDataset()



'''

				#shoe_image = np.array(Image.open(format_list[1]),'f')
				#print shoe_image.shape,shoe_image.dtype,format_list[3]

		#print shoe_image,format_list[3]
	            #shoesImg_nparrayList.append((shoe_image,format_list[3]))
	            #pickle.dump((shoe_image,format_list[3]),pkl)
	            			
from PIL import Image
>>> f=open("footwear.jpg","rb")
>>> im=Image.open("footwear.jpg")
>>> im.thumbnail((64,64))#it takes into account aspect ratio,
>>> im.save("abc.jpg")


>>> im=Image.open("footwear.jpg")
>>> im=im.resize((128,128),Image.ANTIALIAS)
>>> im.save("resize128.jpg","JPEG",quality=90)

>>> im.save("resize_subsample0_128.jpg","JPEG",subsampling=0,quality=100)




http://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
                	x,y,z = shoe_image.shape #width,height,color
                	for i in xrange(x):
                		for j in xrange(y):
                			for k in xrange(z):
                				print shoe_image[i][j],format_list[3]
                	
                	              	
                	for j in xrange(3):
	                	print shoe_image[...,j],format_list[3]
	                	minval = shoe_image[...,j].min()
	                	maxval = shoe_image[...,j].max()
	                	print minval,maxval
	                	newmin=0.0
	                	newmax=1.0
	                	if minval != maxval:
	                		shoe_image[...,j]-=minval
	                		shoe_image[...,j]*=(255.0/(maxval-minval))
	                	#shoe_image+=newmin
	                	print shoe_image[...,j],format_list[3]
'''

