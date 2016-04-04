from PIL import Image
import cPickle as pickle
import numpy as np
import os
def convert():
    l1=["sandals-floaters","slippers-flipflop","sneakers","sports","formal","loafers","ethic","mens-boot"]
    l2=["flats","heels","ballerinas","womens-canvas","womens-boot"]

    path=os.getcwd()
    pathmen=os.path.join(path,"mens")
    
    print path
    folders=os.listdir(path)

    dataset=[]
    labels=[]
    path_name=[]
    #return 
    label_id=0
    for folder in l1:
        print folder,label_id
        if os.path.isdir(pathmen+"/"+folder):   
            #print len(os.listdir(pathmen+"/"+folder))
            img_names=os.listdir(pathmen+"/"+folder)
            #print img_names[:5]
            count=0
            sk=0
            for fn in img_names:
                #print fn
                try:
                    shoe_image = Image.open(pathmen+"/"+folder+"/"+fn)              
                    shoe_image=shoe_image.resize((32,32),Image.ANTIALIAS)
                    shoe_image=np.array(shoe_image,dtype="f")
                    if shoe_image.shape != (32,32,3):
                        print shoe_image.shape
                        continue
                    dataset.append(shoe_image)
                    labels.append(label_id)
                    path_name.append(pathmen+"/"+folder+"/"+fn)
                    count+=1
                    if count>=2500:
                        break
                except:
                    print fn + "skipped...\n"
                    sk+=1   
        
            print sk
            label_id+=1


    pathwomen=os.path.join(path,"womens")
    for folder in l2:
        print folder,label_id
        if os.path.isdir(pathwomen+"/"+folder): 
            #print len(os.listdir(path+"/"+folder))
            img_names=os.listdir(pathwomen+"/"+folder)
            #print img_names[:5]
            count=0
            sk=0

            for fn in img_names:
                #print fn
                try:
                    shoe_image = Image.open(pathwomen + "/" + folder + "/" + fn)                
                    shoe_image = shoe_image.resize((32, 32), Image.ANTIALIAS)
                    shoe_image = np.array(shoe_image, dtype="f")
                    if shoe_image.shape != (32, 32, 3):
                        print shoe_image.shape
                        continue
                    dataset.append(shoe_image)
                    labels.append(label_id)
                    path_name.append(pathwomen + "/" + folder + "/" + fn)
                    count += 1
                    if count >= 2500:
                        break
                except:
                    print fn + "skipped...\n"
                    sk += 1   
        
            print sk
            label_id += 1

    print len(dataset), len(labels)      

    data = dataset, labels, path_name

    pkl = open("shoes_dataset32_label_pathnames.pickle", "wb")
    pickle.dump(data, pkl, pickle.HIGHEST_PROTOCOL)#for binary file protocol>=1
    pkl.close()











convert()