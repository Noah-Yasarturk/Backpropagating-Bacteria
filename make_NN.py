# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 01:31:32 2018

@author: nyasa
"""


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image



def main():
    #Path to where testing set is stored
    fTest = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/rs_set/testing_images_rs/'
    #Path to where training set is stored
    fTrain = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/rs_set/training_images_rs/'
    
    
    #Load data
    trn_imRAs = []
    tst_imRAs = []
    #Index of image array will correspond to index of classification array
    trn_cls = [] 
    tst_cls =[]
    get_data(fTrain, fTest, trn_imRAs, tst_imRAs, trn_cls, tst_cls)

    
    #Import testing and training images to arrays
    print('Getting data....')
    
    #trn_cls and tst_cls has the real strains of each image from 1 to 344
    #Convert to values from real strain to range from 0 to 64, the labels
    trn_lbls = cls_to_lbl(trn_cls)
    tst_lbls = cls_to_lbl(tst_cls)
    
    
    ###print(sorted(set(tst_cls)))
    print('The length of the image array is '+str(len(trn_imRAs))+'.')
    print('The number of strains we are classifying is '+str(len(tst_imRAs)/2)+'.')
    ###print(type(trn_imRAs[0]))
    
    #Convert list of numpy arrays to numpy array
    trn_imRAs = np.array(trn_imRAs)
    tst_imRAs = np.array(tst_imRAs)    
    
    #Cast ints of img array to floats
    for i in range(len(trn_imRAs)):
        trn_imRAs[i].astype(float)
    for i in range(len(tst_imRAs)):
        tst_imRAs[i].astype(float)   
              
    #Check data
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(trn_imRAs[i], cmap=plt.cm.binary)
        plt.xlabel(trn_cls[i])
    
    
    #Create model instance; inputs are pixel size and neurons
    model = create_model(365, 128)
    
    
    #Train the model
    print('Training model....')
    model.fit(trn_imRAs, trn_lbls, epochs=10)
    
    #Evaluate model
    test_loss, test_acc = model.evaluate(tst_imRAs, tst_lbls)
    print('Test accuracy:', test_acc)
    
    #Make predictions
    predictions = model.predict(tst_imRAs)
    print(np.argmax(predictions[0]))
    #Get highest confidence value for this image
    
    #Check test label to see if we're right
    



#Methods
def create_model(pix, neurs):
    print('Creating model....')
    model = keras.Sequential([
    #Flatten layer condenses arrays
    keras.layers.Flatten(input_shape=(pix, pix,3)),
    #Dense layer 
    keras.layers.Dense(neurs, activation=tf.nn.relu),
    #Final layer predicts
    keras.layers.Dense(65, activation=tf.nn.softmax)
    ])
    
    #Compile model
    print('Copiling model...')
    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return(model)
    
    
def create_model_conv(pix, neurs):
    print('Creating model....')
    model = keras.Sequential([
            #Layers here
            
            ])
    #Compile model
    print('Copiling model...')
    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return(model)
    
def get_data(fTrain, fTest, trn_imRAs, tst_imRAs, trn_cls, tst_cls):
    nothing = 0
    for strn_num in range(1,345):
        for strn_copy in range(1,5):
            #Handle hidious strain 13
            if (strn_num == 13):
                nothing +=1
            else:             
                thisBact_c =  bact_img(strn_num,strn_copy,1)
                thisBact_s = bact_img(strn_num,strn_copy,2)
                #Check if the file exists
                aTestImg_c = Path(fTest+thisBact_c)
                aTestImg_s = Path(fTest+thisBact_s)
                aTrainImg_c = Path(fTrain +thisBact_c)
                aTrainImg_s = Path(fTrain +thisBact_s)
                if (aTestImg_c.is_file()==True) and  (aTestImg_s.is_file()==True):
                    #Add circle image array
                    img_c = Image.open(aTestImg_c)
                    ac = np.array(img_c)
                    tst_imRAs.append(ac)
                    tst_cls.append(int(bact2strn(thisBact_c)))
                    #Add Sobel image array
                    img_s = Image.open(aTestImg_s)
                    a_s = np.array(img_s)
                    tst_imRAs.append(a_s)
                    tst_cls.append(int(bact2strn(thisBact_s)))
                    
        
                if (aTrainImg_c.is_file()==True) and (aTrainImg_s.is_file()==True):
                    #Add circle image array
                    img_c = Image.open(aTrainImg_c)
                    ac = np.array(img_c)
                    trn_imRAs.append(ac)
                    trn_cls.append(int(bact2strn(thisBact_c)))
                    #Add Sobel image array
                    img_s = Image.open(aTrainImg_s)
                    a_s = np.array(img_s)
                    trn_imRAs.append(a_s)
                    trn_cls.append(int(bact2strn(thisBact_s)))


def cls_to_lbl(t_cls):
    outRA = []
    strains = [1, 2, 3, 5, 9, 14, 23, 26, 29, 32, 33, 38, 41, 45, 47, 50, 52, 69, 73,
               78, 84, 86, 90, 94, 109, 113, 120, 122, 124, 125, 126, 127, 128, 129,
               132, 133, 134, 148, 149, 158, 159, 169, 174, 175, 210, 226, 230, 231,
               232, 237, 259, 276, 293, 298, 304, 313, 317, 329, 331, 333, 334, 337,
               338, 340, 344]
    for i in range(len(t_cls)):
        val = t_cls[i]
        lbl = strains.index(val)
        outRA.append(lbl)
    return(outRA)



def plot_image(i, predictions_array, true_label, img, trn_cls):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(trn_cls[predicted_label],
                                100*np.max(predictions_array),
                                trn_cls[true_label]),
                                color=color)



def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)
    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)
    return 1 if brightness == 255 else brightness / scale   
    
def bact_img(strain, copy_number, v):
    #v=0 is nothing, 1 is circle, 2 is Sobel
    if (v==0):
        fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number)+'.jpg')
    if (v==1):
        fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number)+'-c.jpg')
    if (v==2):
        fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number)+'-s.jpg')
    return(fileString)     
    
def make3num(num):
    outS = ''
    if num <10:
        outS = ('00'+str(num))
    if num <100:
        outS = ('0'+str(num))
    else:
        outS = str(num)
    return(outS+'/')

def bact2strn(bact_img_output):  
#Returns the strain number from image file        
    #Strain is the first number, coming after a dash and before a dash
    spl = bact_img_output.split('_',1)
    spl2 = spl[0].split('-',1)
    return(spl2[1]) 
    
if __name__ == "__main__":
    main() 
