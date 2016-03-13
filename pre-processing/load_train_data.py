import numpy as np
import pickle

def load_dataset(f):
		#op = open("../train_images/shoesImg_nparray.pkl","rb")
		op = open(f,"rb")
		listOftuple = pickle.load(op)
		total=len(listOftuple)#total no of images=3511
		op.close
		#return listOftuple
		training_set=[]
		validation_set=[] 
		test_set=[]
		i = 0
		#total no of classes = 70
		#class label ids are from 500 to 569
		#70*5 for validation 
		# 70*5 for testing
		#and remaining 2811 for training 

		while i<total:
			for j in xrange(5):#picking top5 from each class as validation
				validation_set.append(listOftuple[i])
				i+=1
			for j in xrange(5):#picking next 5 for testing from each class
				test_set.append(listOftuple[i])
				i+=1
				
			while  i<total and int(listOftuple[i][1])==int(listOftuple[i-1][1]) :
				training_set.append(listOftuple[i])
				i+=1
		print len(training_set),len(validation_set),len(test_set)				 
		return training_set,validation_set,test_set

		'''
		from load_train_data import *
		f="../train_images/shoesImg_nparray64b.pickle"
		train,validation,test=load_dataset(f)


		'''

#count no of images in each class	
def countClasses(fl):
	#f=open("../train_images/release/train_annos.txt","r")
	f = open(fl,"r")
	classCounter = [0]*70
	print classCounter

	for i in f:
		format_list = i.split(",")
		x=int(format_list[0])
		if x>=45542:
			#print format_list[1],format_list[3]			
			classCounter[int(format_list[3])-500]+=1
	print	classCounter
	top20 = sorted(enumerate(classCounter),key=lambda x:x[1])[-20:]
	print top20
	print [sum(x) for x in zip(*top20)]#sum of top20 images dataset



