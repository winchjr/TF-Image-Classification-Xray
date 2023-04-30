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
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
from datetime import datetime
import pandas as pd
from pathlib import Path

print('=====BEGIN PREDICTION=====')
print('Now Loading Models...')

#load all the models in
g1_ate_cardio = load_model('/path/to/model/1')
g2_cons_edem = load_model('/path/to/model/2')
g3_eff_emph = load_model('/path/to/model/3')
g4_inf_mass = load_model('/path/to/model/4')
g5_nod_pleu_pneu = load_model('/path/to/model/5')
#test_model = load_model('/home/jrwinch1/Desktop/NIH2/scripts/Models/test4')

#read in the extra data about each image, its patient, age, etc
all_data = pd.read_csv('/path/to/Data_Entry_2017.csv')

save_csv = '/path/to//Data_Entry_2017_With_Predictions'

#set root data folders for images
root_train_folder = '/path/to/TrainData'
root_test_folder = '/path/to/TestData'

#load in the disease names
#diseases = ["Atelectasis","Effusion","Infiltration","Mass","Nodule","Pneumothorax","Consolidation","Pleural_Thickening","No_Finding","Cardiomegaly","Edema","Emphysema"]
diseases = ["Atelectasis"]

img_size = 256

#adding columns for prediction and the disease that this prediction was made 
all_data['correct_prediction'] = pd.Series(dtype='int')
all_data['disease_test'] = pd.Series(dtype='string')

#this makes an empty data frame that has the same columns as our all_data does, so we can concat to it in the for loop
disease_data = pd.DataFrame(columns = all_data.columns)

print('Models Successfully Loaded!')
print('Now Making Predictions...')

#for each disease,
for disease in diseases:
    #true label number. unfortunately this depends on which model it was for... its set to a bad number here, to throw red flags if it tries to use this
    true_label_num = 5

    #assign the current group
    if disease == 'Atelectasis' or disease=='Cardiomegaly':
        current_model = g1_ate_cardio
        current_group = 1

        if disease == 'Atelectasis':
            true_label_num = 0
        else:
            true_label_num = 0
    if disease == 'Consolidation' or disease=='Edema':
        current_model = g2_cons_edem
        current_group = 2

        if disease == 'Consolidation':
            true_label_num = 1
        else:
            true_label_num = 0
    if disease == 'Effusion' or disease=='Emphysema':
        current_model = g3_eff_emph
        current_group = 3

        if disease == 'Effusion':
            true_label_num = 1
        else:

            true_label_num = 0
    if disease == 'Infiltration' or disease=='Mass' or disease=='No_Finding': #test the no findings ones here
        current_model = g4_inf_mass
       # current_model = test_model
        current_group = 4
        if disease == 'Infiltration':
            true_label_num = 1
        elif disease =='Mass':
            true_label_num = 0
        else:
            true_label_num = 2
    if disease == 'Nodule' or  disease=='Pleural_Thickening' or disease =='Pneumothorax':
        current_model = g5_nod_pleu_pneu
        current_group = 5
        if disease == 'Pleural_Thickening':
            true_label_num = 1
        elif disease=='Nodule':
            true_label_num = 2
        else:
            true_label_num = 0

    print('The true label number is: ', true_label_num)

    #make the train and test image folder locations
    train_folder = os.path.join(root_train_folder,str(current_group))
    test_folder = os.path.join(root_test_folder, str(current_group))

    #add the disease on to the file path too, ie /TrainData/<group#>/<disease>
    train_folder = os.path.join(train_folder,disease)
    test_folder = os.path.join(test_folder,disease)

    #slice the data that contains the disease as a description in the Finding Labels
    current_disease_data = all_data[all_data['Finding Labels'].str.contains(disease)]

    #assign the current disease to its own column, so we know which disease this prediction was made for
    current_disease_data.assign(disease_test=disease)
    #print(type(current_disease_data))

    #create function that gets the data from the folders
    def get_data(data_dir):

        print('Now loading data from',data_dir)
        data = [] 
        incorrect_count = 0
        try_this_many = 0
        for img in os.listdir(data_dir):
                try_this_many = try_this_many  + 1
                if try_this_many < 1000000:
                    img_arr = cv2.imread(os.path.join(data_dir, img)) 
                    resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size 
                    resized_arr = np.array(resized_arr).astype('float32')/255 #you rescaled in the original; do the same here
                    resized_arr = np.expand_dims(resized_arr,0) #have to do this for predicting single images

                    prediction = current_model.predict(resized_arr)


              #if the image is correctly predicted and matches the true_label_num, then put a 1 in its corresponding column for that image row

                    if np.argmax(prediction[0])== true_label_num:
                        current_disease_data.loc[current_disease_data['Image Index']==img, 'correct_prediction'] = str(1)
                        current_disease_data.loc[current_disease_data['Image Index']==img,'disease_test'] = disease


                    else:
                        incorrect_count = incorrect_count + 1
                        current_disease_data.loc[current_disease_data['Image Index']==img, 'correct_prediction'] = str(0)
                        current_disease_data.loc[current_disease_data['Image Index']==img,'disease_test'] = disease


        print('I found',incorrect_count,'bad guesses for',disease)


    print('Now making guesses for', disease)
    #retrieve the data, store as train and test
    get_data(train_folder)
    get_data(test_folder)

    current_save_csv = save_csv + '_' + disease + '.csv'
    print(current_save_csv)
    current_disease_data.to_csv(current_save_csv, index=False)

print('Done!')
