# Welcome

Based on the VGG16 example on Keras Blog, some tutorials, adjustments, this
script can function as a document classifier, based on the 'shape'of the
header of an image, It takes the upper third of a page, to train the model 
and to evaluate it.
    
     ______________
    | * * * * * *  |
    |  THIS AREA   |
    |_*_*_*_*_*_* _|
    |              |
    |              |
    |______________|
    |              |
    |              |
    |______________|
        
It was trained with over 40000 documents (Invoices, Credit Notes, Proformas and
other types of documents)

To train the network from Scratch:
    
1.- Define a root path, ie, "E:/drm_classes15"
2.- Inside of E:/drm_classes15, create a directory for each class (type) of 
    document you want the DNN to learn to classify.
    
    For example:
        
        E:/drm_classes15
            /Invoices
            /CreditNotes
            /Contracts
            /Formulars
            /Others
    
    Try to provide as many as possible documents for each category, let's say,
    a minimum of 5000 of each kind. Be careful to not over biase the network, 
    don't put many more samples of one kind, and much less from others, otherwise
    the training will be skewed and your accuracy will be bad.
    
3.- Evaluate the model, check the Confussion Matrix and Losses
4.- If needed, play around with the network definition, or hyperparameters.
5.- Use the saved model and load it for production use

Time to train:
    Using Tensorflow with GPU and ~40000 training samples, takes about 3 hours,
    on a CPU Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz (6 cores), 32Gb of RAM 
    and a GPU NVIDIA 980 (The GPU is the one helpful here).

# Using it:

Just pull your python editor and run it.
This was enough for me, but you can add your own `main` function and parametrize it.

# Training:
Feature Map Layer 1:
![alt text](https://github.com/coloboxp/DocumentClassificationByShape/raw/master/img/fml1.jpg "Feature Map Layer 1")
Feature Map Layer 1, Filter 0:
![alt text](https://github.com/coloboxp/DocumentClassificationByShape/raw/master/img/fml10.jpg "Feature Map Layer 1: Filter 0")

Feature Map Layer 2:
![alt text](https://github.com/coloboxp/DocumentClassificationByShape/raw/master/img/fml2.jpg "Feature Map Layer 2")
Feature Map Layer 2, Filter 0:
![alt text](https://github.com/coloboxp/DocumentClassificationByShape/raw/master/img/fml20.jpg "Feature Map Layer 2: Filter 0")

Feature Map Layer 3:
![alt text](https://github.com/coloboxp/DocumentClassificationByShape/raw/master/img/fml3.jpg "Feature Map Layer 3")
Feature Map Layer 3, Filter 0:
![alt text](https://github.com/coloboxp/DocumentClassificationByShape/raw/master/img/fml30.jpg "Feature Map Layer 3: Filter 0")

Feature Map Layer 4:
![alt text](https://github.com/coloboxp/DocumentClassificationByShape/raw/master/img/fml4.jpg "Feature Map Layer 4")
Feature Map Layer 4, Filter 0:
![alt text](https://github.com/coloboxp/DocumentClassificationByShape/raw/master/img/fml40.jpg "Feature Map Layer 4: Filter 0")

Confussion Matrix:
The results after evaluating training samples with the model, using only 3 classes.
![alt text](https://github.com/coloboxp/DocumentClassificationByShape/raw/master/img/cm.png "Confussion Matrix")
