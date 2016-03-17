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
		training_label=[]
		validation_set=[]
		validation_label=[] 
		test_set=[]
		test_label=[]
		i = 0
		#listOftuple contains list of (numpy_array,class_label)
		#total no of classes = 70
		#class label ids are from 500 to 569
		#70*5 for validation 
		# 70*5 for testing
		#and remaining 2811 for training 


		


		while i<total:
			#print listOftuple[i][0].shape
			#return listOftuple
			
			j=0	
			while j<5:#picking top5 from each class as validation
				if listOftuple[i][0].shape != (64,64,3):
					print 'skip image no',i,listOftuple[i][0].shape
					i=i+1
					continue
				validation_set.append(listOftuple[i][0])
				validation_label.append(int(listOftuple[i][1]))
				i+=1
				j+=1



			j=0	
			while j<5:#picking next 5 for testing from each class
				if listOftuple[i][0].shape != (64,64,3):
					print 'skip image no',i,listOftuple[i][0].shape
					i=i+1
					continue
				test_set.append(listOftuple[i][0])
				test_label.append(int(listOftuple[i][1]))
				i+=1
				j+=1
			while  i<total and int(listOftuple[i][1])==int(listOftuple[i-1][1]) :
				if listOftuple[i][0].shape != (64,64,3):
					print 'skip image no',i,listOftuple[i][0].shape
					i=i+1
					continue
				training_set.append(listOftuple[i][0])
				training_label.append(int(listOftuple[i][1]))
				i+=1
		print 'training, validation, testing'
		print len(training_set),len(validation_set),len(test_set)	
		#np.array(training_set)
		#return training_set
		return np.array(training_set),np.array(training_label),np.array(validation_set),np.array(validation_label),np.array(test_set),np.array(test_label)

		'''
		from load_train_data import *
		f="../train_images/shoesImg_nparray64b.pickle"
		train,validation,test=load_dataset(f)

while i<total:
			#print listOftuple[i][0].shape
			#return listOftuple
			if listOftuple[i][0].shape != (64,64,3):
				print 'skip image no',i,listOftuple[i][0].shape
				i=i+1
				continue
			i+=1	
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



