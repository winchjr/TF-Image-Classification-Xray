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
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from sklearn.svm import SVC
import tensorflow_addons as tfa

begin_time = datetime.now()

### BEGIN CONFIG ###

path = "/path/to/data"
checkpoint_folder = "/path/to/checkpoint"
checkpoints_csvlocation = "/path/to/checkpoints.csv"
save_model_location = "/path/to/save/models"

#train split, where (1-train_split)/2 = valid, test size
train_split = 0.90

img_size = 256

#as long as its greater than the max size of training data then its ok
buffer_shuffle_size = 100000

#batch size: how many images per dataset
tf_batch_size = 8196

#setting dropout level
dropout_levels = [.25, .35, .45]

#do we want to save the current_model when done training it?
save_model = False

#delta to be used for early stopping testing
deltas = [0.01]

#optimizer list to test
optimizers = ['Adam']

#number of neurons should be between num_labels and total input size. 
neurons = [128, 256]

#batch size can only be as big as the smallest training set folder. For instance, if 1 folder has only 20 images, the largest batch size can only go to 20.
batch_sizes = [16, 32]

verbosity_level = 1

loop_times = 8

#squelch tf output in terminal. options are: info, warning, error, none. none squelches all, info includes all.
tf.get_logger().setLevel('FATAL')

### END CONFIG ###

#define feature map
feature_map = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_id': tf.io.FixedLenFeature([], tf.string),
    'No Finding': tf.io.FixedLenFeature([], tf.int64),
    'Atelectasis': tf.io.FixedLenFeature([], tf.int64),
    'Consolidation': tf.io.FixedLenFeature([], tf.int64),
    'Infiltration': tf.io.FixedLenFeature([], tf.int64),
    'Pneumothorax': tf.io.FixedLenFeature([], tf.int64),
    'Edema': tf.io.FixedLenFeature([], tf.int64),
    'Emphysema': tf.io.FixedLenFeature([], tf.int64),
    'Fibrosis': tf.io.FixedLenFeature([], tf.int64),
    'Effusion': tf.io.FixedLenFeature([], tf.int64),
    'Pneumonia': tf.io.FixedLenFeature([], tf.int64),
    'Pleural_Thickening': tf.io.FixedLenFeature([], tf.int64),
    'Cardiomegaly': tf.io.FixedLenFeature([], tf.int64),
    'Nodule': tf.io.FixedLenFeature([], tf.int64),
    'Mass': tf.io.FixedLenFeature([], tf.int64),
    'Hernia': tf.io.FixedLenFeature([], tf.int64)
    }





def tfr_decoder(path, shuffle=False):
    def image_decoder(data):
        example = tf.io.parse_single_example(data, feature_map) 
        image = example['image']
        image = tf.io.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, img_size, img_size)
        image.set_shape([img_size,img_size,3])
        image = image/255.
        #applying augmentation, including random flips vert/horiz, and random rotation, but only to training data
        #image = tf.image.random_flip_left_right(image)
        #image = tf.image.random_flip_up_down(image)
        #image= tfa.image.rotate(image, random.randrange(90))

        #print([label for label in sorted(list(example.keys())) if label!='image' and label!='image_id'])
        labels = [tf.cast(example[x], tf.float32) for x in sorted(list(example.keys())) if x!='image_id' and x!='image']

        return image, labels

    #splitting training data
    data_list = [os.path.join(path,x) for x in os.listdir(path)]
    split = int(len(data_list)*train_split)
    train_data, val_data = data_list[:split], data_list[split:]

    trainds = tf.data.TFRecordDataset(train_data)
    trainds = trainds.map(image_decoder)

    valds = tf.data.TFRecordDataset(val_data)
    valds = valds.map(image_decoder)

    if shuffle:
        trainds = trainds.shuffle(buffer_shuffle_size)
        valds = valds.shuffle(buffer_shuffle_size*(1-train_split))

    trainds = trainds.batch(tf_batch_size, drop_remainder=True)
    valds = valds.batch(tf_batch_size, drop_remainder=True)

    valds = valds.repeat(9)

    return trainds, valds

train, valid = tfr_decoder(path)





def create_model():
    current_model = tf.keras.Sequential([
        Conv2D(batch_size,(3,3),activation='relu',input_shape=(img_size,img_size,3)),
        MaxPool2D(2,2),
        Flatten(),
        Dense(neuron,activation='relu'),
        Dropout(dropout_level),
        Dense(int(neuron/2),activation='relu'),
        Dropout(dropout_level),
        Dense(15,activation='sigmoid')
  ])
    current_model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(.0001),metrics=['accuracy'])

    return current_model





#fit the model, given some data
def fit_model(current_model, train_x, train_y, valid_x, valid_y, count):

    print("Now Training More Data...")
    #reloading the weights to the latest checkpoint
    if count != 0:
        current_model.load_weights(os.path.join(checkpoint_folder, 'current_model.h5'))

    #fit the model
    current_model.fit(train_x,train_y,epochs=15,batch_size=batch_size,callbacks=[monitor_callback], verbose=verbosity_level)

    current_model.save_weights(os.path.join(checkpoint_folder, 'current_model.h5'))
    print("Now saving weights to", os.path.join(checkpoint_folder, 'current_model.h5'))

    #get the accuracy/loss metrics
    test_loss, test_acc = current_model.evaluate(valid_x,valid_y, verbose=verbosity_level)

    #make prediction
    #pred = current_model.predict(valid_x)

    #metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=15)
    #metric.update_state(valid_y,pred)

    #confusion matrix. gets the result from the MLCM above. use numpy.result() to see the 15*15 grid confusion matrix.
    #cm = metric.result()

    print("Accuracy for this fit: ",test_acc,sep="")
    print("Loss for this fit: ",test_loss,sep="")                               
    #save the model info to csv
    with open(checkpoints_csvlocation, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
        writer.writerow([count] + [begin_time] + [datetime.now()] + [datetime.now()-begin_time]  + [optimizer] + [img_size] + [neuron] + 
                        [batch_size] + [d] + [dropout_level] + [test_acc] + [test_loss])

   # ckpt.step.assign_add(1)
    count = count + 1

    return count






#define train and checkpoint function. This will iterate over all the data using the currently defined net model
def train_and_checkpoint(train, valid):

    #iterator object to go through all the train data
    train_iter = iter(train)
    valid_iter = iter(valid)

    current_model = create_model()

    #create checkpoints, and checkpoint manager
    #ckpt = tf.train.Checkpoint(step=tf.Variable(0), iterator=train_iter, net=current_model)
    #manager = tf.train.CheckpointManager(ckpt, checkpoint_folder, max_to_keep=2)

    #shuffle the data
    #train.shuffle(600)
    #valid.shuffle(600)

    #take 4 elements from the dataset
    #current_train = train.take(4)
    #current_valid = valid.take(4)

    #because we are automatically restoring the best weights, we have to do the first run manually, then enter the loop
    current_train = next(train_iter)
    current_valid = next(valid_iter)

    #unpack data
    train_x, train_y = current_train
    valid_x, valid_y = current_valid

    count = 0       

    #fit the model once, after which we can begin restoring weights
    count = fit_model(current_model, train_x, train_y, valid_x, valid_y,count)

    #save_path = manager.save()
    #print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

    #running this n times
    for _ in range(loop_times):

        #get the next training data
        current_train = next(train_iter)

        #pick a random validation data to use to validate against
        current_valid = next(valid_iter)

        #shuffle the data
        #train.shuffle(600)
        #valid.shuffle(600)

        #take 4 elements from the dataset
        #current_train = train.take(4)
        #current_valid = valid.take(4)

        #unpack the images (x) and labels (y)
        train_x, train_y = current_train
        valid_x, valid_y = current_valid

        #create new model 
        current_model = create_model()

        #select random valid dataset to work with here
        count = fit_model(current_model, train_x, train_y, valid_x, valid_y, count)



        #save_path = manager.save()
        #print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        #current_model.load_weights(tf.train.latest_checkpoint(checkpoint_folder))

        #clears the current model
        tf.keras.backend.clear_session() 

    #if we are going to save the model when done training, do it here
    if save_model:
        current_model.save(os.path.join(save_model_location,count))






#grid search over all these model parameters
for optimizer in optimizers:
    for neuron in neurons:
        for batch_size in batch_sizes:
            for dropout_level in dropout_levels:
                for d in deltas:
                    print("========MODEL ARCHITECTURE========")
                    print("Optimizer: ", optimizer,sep="")
                    print("Neurons: ", neuron,sep="")
                    print("Batch Size: ", batch_size,sep="")
                    print("Dropout Level: ", dropout_level,sep="")
                    print("Delta Value: ",d,sep="")
                    print("============================")

                    #create stopping params
                    monitor_callback = EarlyStopping(monitor='val_loss', 
                                                     patience=3, 
                                                     min_delta=d)

                    # Create a callback that saves the model's weights
                    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_folder,
                                                                       save_weights_only=True,
                                                                       verbose=0,
                                                                       save_freq=1*batch_size)


                    #begin training this current model
                    print("Now Starting Next Training And Checkpointing Run...")
                    train_and_checkpoint(train, valid)
