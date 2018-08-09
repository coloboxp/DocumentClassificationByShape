# -*- coding: utf-8 -*-
"""
Spyder Editor

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

"""
import datetime
import itertools
import os

from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import cv2

K.clear_session()
K.set_image_dim_ordering("tf") #Use TensorFlow

DATA_PATH = "E:/drm_classes15"
#os.chdir("E:\drm_classes")
DATA_DIR_LIST = os.listdir(DATA_PATH)

#defining the dimensions of the resized image in similar proportion as the one third
IMG_COLS = 362
IMG_ROWS = 170
NUM_CHANNEL = 1 #black and white
NUM_EPOCH = 15

IMG_DATA_LIST = []
IMG_TYPES = []

START = 0
OFFSET = 0
CLASS_ID = 0

def show_timestamp(text):
    '''
    Acts a log on on the console, including timestamp
    '''
    print(datetime.datetime.now().time().strftime("%H:%M:%S.%f") + " " + str(text))

def get_image(path):
    '''
    Loads and pre-processes an image, for training and evaluation
    '''
    input_img = cv2.imread(path)
    height, _, _ = input_img.shape
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # Here I'm trying just to get the upper third part of the document
    input_img = input_img[0:int(height/3), :] 
    return cv2.resize(input_img, (IMG_COLS, IMG_ROWS))

def plot_results(hist):
    '''
    Plots the graphs for loss and accuracy for training and validation
    ''' 
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    train_acc = hist.history["acc"]
    val_acc = hist.history["val_acc"]
    x_c = range(NUM_EPOCH)

    #Visualizing the loss and the accuracy
    plt.figure(1, figsize=(7, 5))
    plt.plot(x_c, train_loss)
    plt.plot(x_c, val_loss)
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.title("train_loss vs val_loss")
    plt.grid(True)
    plt.legend(["train", "val"])
    plt.style.use(["classic"])

    plt.figure(2, figsize=(7, 5))
    plt.plot(x_c, train_acc)
    plt.plot(x_c, val_acc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("train_acc vs val_acc")
    plt.grid(True)
    plt.legend(["train", "val"], loc=4)
    plt.style.use(["classic"])

def build_model(input_shape):
    ''' 
    Builds our variant of VGG16
    '''
    model = Sequential()

    #32 filters of 3x3, using same border model and the input shape of the model
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))#0.5

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        show_timestamp("Normalized confusion matrix")
    else:
        show_timestamp("Confusion matrix, without normalization")

    show_timestamp(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

# Visualizing the intermediate layer
def get_featuremaps(model_tmp, layer_idx, X_batch):
    '''
    Extracts the features of a given layer
    '''
    
    get_activations = K.function([model_tmp.layers[0].input,
                                  K.learning_phase()],
                                 [model_tmp.layers[layer_idx].output,])
    activations = get_activations([X_batch, 0])
    return activations

def reshape_for_backend_framework(img_data_temp):
    ''' 
    Process the image according on the backend, if its TF or Theano
    as they use different ordering for color channels.
    '''
    
    show_timestamp("Adapting the structure to be compatible with: " + K.image_dim_ordering())
    if NUM_CHANNEL == 1:
        if K.image_dim_ordering == "th":
            #if using theano as backend, we then have to add the column at index 1
            img_data_temp = np.expand_dims(img_data_temp, axis=1)
            #just to confirm the shape is correct
            show_timestamp(img_data.shape)
        else:
            #if using tensorflow as backend, then the channel's dimension is added at the end
            img_data_temp = np.expand_dims(img_data_temp, axis=4)
            #just to confirm the shape is correct
            show_timestamp(img_data_temp.shape)
    else:
        if K.image_dim_ordering == "th":
            #if using TF backend, and loading a RGB image, then just swap
            #the dimension from index 3 to index 1
            img_data_temp = np.rollaxis(img_data_temp, 3, 1)
            #just to confirm the shape is correct
            show_timestamp(img_data_temp.shape)
    return img_data_temp

def show_feature_map_at_layer(layer_num):
    '''
    Shows the network's learned features at a given layer
    '''
    show_timestamp("Feature map for layer    : " + str(layer_num))
    filter_num = 0
    activations = get_featuremaps(model, int(layer_num), TEST_IMAGE)

    show_timestamp("Activations shape        : " + str(np.shape(activations)))
    feature_maps = activations[0][0]

    show_timestamp("Feature maps shape       : " + str(np.shape(feature_maps)))
    if K.image_dim_ordering() == "th":
        feature_maps = np.rollaxis((np.rollaxis(feature_maps, 2, 0)), 2, 0)
        show_timestamp("Feature maps shape       : " + str(feature_maps.shape))

    fig = plt.figure(figsize=(16, 16))
    plt.imshow(feature_maps[:, :, filter_num], cmap="gray")
    plt.savefig("featuremaps-layer-{}".format(layer_num) +
                "-filternum-{}".format(filter_num)+".jpg")

    num_of_featuremaps = feature_maps.shape[2]
    fig = plt.figure(figsize=(16, 16))
    plt.title("Featuremaps-Layer-{}".format(layer_num))
    subplot_num = int(np.ceil(np.sqrt(num_of_featuremaps)))
    for i in range(int(num_of_featuremaps)):
        a_x = fig.add_subplot(subplot_num, subplot_num, i+1)
        a_x.imshow(feature_maps[:, :, i], cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig("Featuremaps-Layer-{}".format(layer_num) + ".jpg")

img_data = None 
IMG_TYPES = []

show_timestamp("Checking if we have precomputed images")
#it might take some gigabytes if the training is too big
if os.path.exists("nparray.npy"):
    show_timestamp("Loading precomputed numpy array")
    for dataset in DATA_DIR_LIST:
        img_list = os.listdir(DATA_PATH + "/" + dataset)
        show_timestamp("Loading images from dataset: " + dataset)
        for img in img_list:
            if img.endswith(".png"):
                OFFSET = OFFSET + 1

        IMG_TYPES.append((dataset, CLASS_ID, START, OFFSET+ START))
        START = START + OFFSET
        OFFSET = 0
        CLASS_ID = CLASS_ID + 1
        show_timestamp("Items loaded " + str(len(img_list)))
    img_data = np.load("nparray.npy")
else:
    show_timestamp("Loading images from disk")
    for dataset in DATA_DIR_LIST:
        img_list = os.listdir(DATA_PATH + "/" + dataset)

        show_timestamp("Loading images from dataset: " + dataset)
        for img in img_list:
            if img.endswith(".png"):
                input_img_resize = get_image(DATA_PATH + "/" + dataset + "/"+ img)
                IMG_DATA_LIST.append(input_img_resize)
                OFFSET = OFFSET + 1

        IMG_TYPES.append((dataset, CLASS_ID, START, OFFSET + START))
        START = START + OFFSET
        OFFSET = 0
        CLASS_ID = CLASS_ID + 1
        show_timestamp("Items loaded :" + str(len(IMG_DATA_LIST)))

    show_timestamp("Converting into array")
    img_data = np.array(IMG_DATA_LIST)
    img_data = img_data.astype("float32")
    show_timestamp("Normalizing")
    img_data /= 255
    np.save("nparray.npy", img_data)

num_classes = len(set(IMG_TYPES))

show_timestamp("Going to classiffy: " + str(num_classes) + " classes")
show_timestamp("Image Types       : " + str(IMG_TYPES))

show_timestamp("Image data shape  : " + str(img_data.shape))

img_data = reshape_for_backend_framework(img_data)

num_of_samples = img_data.shape[0] #we take the amount of items on the first dimension
labels = np.ones((num_of_samples,), dtype="int64")
names = []

show_timestamp("Generating class ranges")

for cname, cid, cstart, cend in IMG_TYPES:
    labels[cstart:cend] = cid
    names.append(cname)

show_timestamp("Labels: " + str(labels))
show_timestamp("Names: " + str(names))

show_timestamp("Converting into categorical (into one hot enconding)")
Y = np_utils.to_categorical(labels, num_classes)

show_timestamp("Shuffling the dataset")
x, y = shuffle(img_data, Y, random_state=2)

show_timestamp("Split the dataset into train and test")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

show_timestamp("Input shape: " + str(img_data[0].shape))

show_timestamp("Creating the DNN")

model = build_model(img_data[0].shape)
hist = model.fit(X_train,
                 y_train,
                 batch_size=8,
                 epochs=NUM_EPOCH,
                 verbose=1,
                 validation_data=(X_test, y_test))

plot_results(hist)

# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)
show_timestamp("Test Loss                : " + str(score[0]))
show_timestamp("Test accuracy            : " + str(score[1]))

TEST_IMAGE = X_test[0:1]
show_timestamp("Test image shape         : " + str(TEST_IMAGE.shape))

show_timestamp("Model.predict test_image : " + str(model.predict(TEST_IMAGE)))
show_timestamp("Model.predict_classes    : " + str(model.predict_classes(TEST_IMAGE)))
show_timestamp("y_test                   : " + str(y_test[0:1]))


# Testing a new image

TEST_IMAGE = get_image("E:/drm_classes/FD_D/00222048.png")
TEST_IMAGE = TEST_IMAGE.astype("float32")
TEST_IMAGE /= 255
show_timestamp("Test image shape         : " + str(TEST_IMAGE.shape))

if NUM_CHANNEL == 1:
    if K.image_dim_ordering() == "th":
        TEST_IMAGE = np.expand_dims(TEST_IMAGE, axis=0)
        TEST_IMAGE = np.expand_dims(TEST_IMAGE, axis=0)
        show_timestamp(TEST_IMAGE.shape)
    else:
        TEST_IMAGE = np.expand_dims(TEST_IMAGE, axis=3)
        TEST_IMAGE = np.expand_dims(TEST_IMAGE, axis=0)
        show_timestamp(TEST_IMAGE.shape)
else:
    if K.image_dim_ordering() == "th":
        TEST_IMAGE = np.rollaxis(TEST_IMAGE, 2, 0)
        TEST_IMAGE = np.expand_dims(TEST_IMAGE, axis=0)
        show_timestamp(TEST_IMAGE.shape)
    else:
        TEST_IMAGE = np.expand_dims(TEST_IMAGE, axis=0)
        show_timestamp(TEST_IMAGE.shape)

# Predicting the test image
show_timestamp(model.predict(TEST_IMAGE))
show_timestamp(model.predict_classes(TEST_IMAGE))

#layer_num=3
show_feature_map_at_layer(1)
show_feature_map_at_layer(2)
show_feature_map_at_layer(3)
show_feature_map_at_layer(4)

Y_pred = model.predict(X_test)
show_timestamp("Y_pred                   : " + str(Y_pred))

y_pred = np.argmax(Y_pred, axis=1)
show_timestamp("y_pred                   : " + str(y_pred))

target_names = names

show_timestamp("class report: " + classification_report(
    np.argmax(y_test, axis=1),
    y_pred,
    target_names=target_names))

show_timestamp("conf matrix : " + str(confusion_matrix(np.argmax(y_test, axis=1), y_pred)))

# Compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
np.set_printoptions(precision=2)
plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names, title="Confusion matrix")
plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("model_dm.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_dm.h5")
show_timestamp("Saved model to disk")

# load json and create model
json_file = open("model_dm.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_dm.h5")
show_timestamp("Loaded model from disk")

model.save("model_dm.hdf5")
loaded_model = load_model("model_dm.hdf5")

K.clear_session()
