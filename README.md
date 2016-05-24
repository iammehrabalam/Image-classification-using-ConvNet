# Footwear Image-classification-using-ConvNet And Visually-similar-Recommendation

## Dataset
We created our own dataset which contain 30,000 footwear images sourced from  
Flipkart, Snapdeal and Amazon. We divided our datasets into twelve class labels.

* sandals-floaters
* slippers­flipflop 
* sneakers 
* sports 
* formal 
* loafers 
* ethnic
* mens­boot
* flats 
* heels 
* ballerinas 
* womens­boot 

[Download Dataset](https://www.dropbox.com/sh/uqvjq31b0k65e15/AAAcYcJXRlMJPf32W7AofSBGa?dl=0)

## Reference
http://cs231n.stanford.edu/reports/nealk_final_report.pdf

# Steps to run the Project
1. Create a virtualenv and activate it
  
  virtualenv myvenv 
  
  source myvenv/bin/activate 
2. pip install -r requirement.txt
3. Run topick.py to convert the dataset in pickel format
4. open ipython notebook 
5. create a new notebook and run following commands (shift + enter to run notebook's cell code)
6. from input_output import Image_classification
7. obj =  Image_classification()
8. obj.training()
9. obj.extract_features()

  output of last fully-connected layer
10. obj.testing()  
11. obj.visually_similar()

[Demo](https://github.com/ahsankamal/Image-classification-using-ConvNet/blob/master/Demo.ipynb)
