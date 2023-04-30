import tensorflow as tf
from keras import layers
import numpy as np
import os
import random
import cv2
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
from datetime import datetime
from keras.callbacks import EarlyStopping
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from sklearn.svm import SVC

begin_time = datetime.now()

### BEGIN CONFIG ###

#folder locations
train_folder_root = '/path/to/TrainData'
test_folder_root = '/path/to/TestData'
cnn_csvlocation = "/path/to/CNNresults.csv"

model_save_location = '/path/to/Models'

#max number of images to load for a label
total_max_img = 4000

#train test split value. select the percent of images you want to be in the train
train_split = 0.90

#model params

#create stopping params
es = EarlyStopping(monitor='sparse_categorical_accuracy', min_delta=.005, patience=2)

#optimizer list to test
optimizers = ['Adam', 'RMSprop']

img_size = 256

#number of neurons should be between num_labels and total input size. 
neurons = [img_size/2,img_size]

#batch size can only be as big as the smallest training set folder. For instance, if 1 folder has only 20 images, the largest batch size can only go to 20.
batch_size = [16,32]


### END CONFIG ###

#calc the train test split
train_max_img = total_max_img * train_split
test_max_img = train_max_img * (1-train_split)

for g in range(1,6):

    #the file structure looks like TrainData/1/Disease/image.png. so select the current group, ie g
    current_train_folder = os.path.join(train_folder_root,str(g))
    current_test_folder = os.path.join(test_folder_root,str(g))

    print('Now loading images...')

    #declare labels and image size
    labels = os.listdir(current_train_folder)
    num_labels = len(labels)

    #create function that gets the data from the folders
    def get_data(data_dir, max_img):

        print('Now loading data from',data_dir)
        data = [] 

        for label in labels: 

            print('Now loading label', label)
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)

            count = 0
            for img in os.listdir(path):

                    if count < max_img: #dont get more than the max_img count, which is dependent on whether its a train or test folder
                        img_arr = cv2.imread(os.path.join(path, img)) 
                        resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                        data.append([resized_arr, class_num])
                    count = count + 1

        return np.array(data, dtype=object)

    #retrieve the data, store as train and test
    train = get_data(current_train_folder, train_max_img)
    test = get_data(current_test_folder, test_max_img)

    #organizing the data into train test splits, with x being data, y being label, etc
    train_x = []
    train_y = []
    for features, label in train:
        train_x.append(features)
        train_y.append(label)
    train_x = np.array(train_x).reshape(-1, img_size, img_size, 3)
    train_y = np.array(train_y)

    test_x = []
    test_y = []
    for features, label in test:
        test_x.append(features)
        test_y.append(label)
    test_x = np.array(train_x).reshape(-1, img_size, img_size, 3)
    test_y = np.array(train_y)

    total_images = len(train_x) + len(test_x)

    print('Done loading data...')
    #verify the data is of correct shape, type, etc
    print('For both train and test combined, we are working with',total_images,'images.')
    print('The type of data we are working with is', type(train_x[0]))
    print('The shape of data we are working with is', train_x.shape)
    print('The min and max image value range for training images, respectively:',(np.min(train_x), np.max(train_x)))
    print('The # of labels we are working with:', num_labels)
    print('The min and max image value range for testing images, respectively:',(np.min(test_x), np.max(test_x)))
    print('The full list of labels: ', labels)

    #scaling images appropriately, to be between 0 and 1
    train_x = train_x / 255
    test_x = test_x / 255



    print('=====BEGIN CNN MODEL BUILDING=====')

    for o in optimizers:

        for neuron in neurons:
            for bs in batch_size:
                start = datetime.now()
                print('=====BEGIN NEXT MODEL=====')
                print('=====MODEL PARAMETERS=====')
                print('Optimizer:',o)
                print('Neurons:',neuron)
                print('Batch Size:',bs)
                #setting the seed so we can easily recreate this model if needed
                tf.random.set_seed(150)

                print('=====BEGIN MODEL BUILD/FIT=====')


                #defining model
                current_model=Sequential()
                #adding convolution layer
                current_model.add(Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size,3)))
                #adding pooling layer
                current_model.add(MaxPool2D(2,2))
                current_model.add(Dropout(0.5))
                #adding fully connected layer
                current_model.add(Flatten())
                current_model.add(Dense(neuron,activation='relu'))
                #adding output layer
                current_model.add(Dense(num_labels,activation='softmax'))

                current_model.summary()

                #compiling the model
                current_model.compile(loss='sparse_categorical_crossentropy',optimizer=o,
                              metrics=['sparse_categorical_accuracy'])
                #fitting the model
                current_model.fit(train_x,train_y,epochs=50,
                             batch_size=bs, callbacks=[es],verbose=1)

                #get the accuracy/loss metrics
                test_loss, test_acc = current_model.evaluate(test_x,test_y, verbose=1)

                #make prediction
                pred = current_model.predict(test_x)

                print('=====FINAL ACCURACY=====')
                print(test_acc)

        #saving the model, if needed
                test_str = 'test2' + str(g)
                current_model_save_location = os.path.join(model_save_location, test_str)
                current_model.save(current_model_save_location)

                #write the model accuracy, results, and params to csvlocation
                with open(cnn_csvlocation, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([begin_time] + [datetime.now()] + [datetime.now()-start] + [g] + [labels] + [o]+ [img_size] + [neuron] + [bs] + [test_acc] + [test_loss])
                tf.keras.backend.clear_session() #clears the xisting model so we can start fresh with the next one
                print('=====Finished in ',(datetime.now()-start),'=====',sep="")
                print('=====END CURRENT MODEL=====')
                ###end cnn loop



print('Finished With All Models And All Diseases in', datetime.now()-begin_time)
