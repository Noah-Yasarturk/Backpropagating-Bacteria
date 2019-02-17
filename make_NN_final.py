'''
Final iteration
'''

#save for reproducability
import numpy as np
np.random.seed(1337)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import csv


def main():
    #Path to where testing set is stored
    fTest = 'Generated/test/'
    #Path to where training set is stored
    fTrain = 'Generated/train/'
    
    
    #Load data
    trn_imRAs = []
    tst_imRAs = []
    #Index of image array will correspond to index of classification array
    trn_cls = [] 
    tst_cls =[]
    tst_nms=[]

    
    #Import testing and training images to arrays
    print('Getting data....')
    get_data2(fTrain, fTest, trn_imRAs, tst_imRAs, trn_cls, tst_cls, tst_nms)
    
    
    #trn_cls and tst_cls has the real strains of each image from 1 to 344
    #Convert to values from real strain to range from 0 to 64, the labels
    trn_lbls = cls_to_lbl(trn_cls)
    tst_lbls = cls_to_lbl(tst_cls)
    
    #Convert labels to numpy array
    trn_lbls = np.array(trn_lbls)
    tst_lbls = np.array(tst_lbls)
    
    
    
    ###print(sorted(set(tst_cls)))
    print('The length of the image array is '+str(len(trn_imRAs))+'.')
    print('The number of strains we are classifying is '+str(len(tst_imRAs))+'.')
    ###print(type(trn_imRAs[0]))
    
    #Convert list of numpy arrays to numpy array
    trn_imRAs = np.array(trn_imRAs)
    tst_imRAs = np.array(tst_imRAs)    
    
    #Check that data is accurate
    print('The length of the training numpy array is '+str(len(trn_imRAs)))
    print('The length of the testing numpy array is '+str(len(tst_imRAs)))
    
    
     #Attempt normalization:   
    trn_imRAs = trn_imRAs.astype('float32')
    tst_imRAs = tst_imRAs.astype('float32')
    
    x_mean = trn_imRAs.mean(axis=0)
    trn_imRAs -= x_mean
    x_std = trn_imRAs.std(axis=0)
    trn_imRAs /= x_std + 1e-8  # add a small constant to prevent division by zero
    
    # normalize test data using the mean and std of training data
    tst_imRAs -= x_mean
    tst_imRAs /= x_std + 1e-8

    
    #Create model instance; inputs are pixel size and neurons
    model = create_model(100, 128)
    
    
    #Train the model
    print('Training model....')
    model.fit(trn_imRAs, trn_lbls, epochs= 25) 
    
    #Evaluate model
    test_loss, test_acc = model.evaluate(tst_imRAs, tst_lbls)
    print('Test accuracy:', test_acc)
    
    #Make predictions
    predictions = model.predict(tst_imRAs)
    print(np.argmax(predictions[0]))
    #Get highest confidence value for this image
    
        
    #Get highest confidence value for this image
    hi_con = []
    hi_val=[]
    for p in range(len(predictions)):
        hi_con.append(np.argmax(predictions[p]))
        hi_val.append(max(predictions[p]))
        
    #Check test label to see if we're right
    csv_path = ('predictions.csv')
    pred_strns = lbl_to_cls(hi_con)
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter = ',')
        #Write top row
        csv_writer.writerow(['Test image','Predicted Strain','Correct?', 'Confidence Value',('Overall Accuracy = '+ str(test_acc))])
        for p in range(len(predictions)):
            #Get image file name
            fName = tst_nms[p]
            #Get predicted strain
            predStrn = pred_strns[p]
            #Check if correct
            cor = 'Nope'
            if (tst_lbls[p] == predStrn):
                cor = 'Yes!'
            val = hi_val[p]
            csv_writer.writerow([fName, predStrn, cor, val])

#Methods
def create_model(pix, neurs):
    #Optimizers
    adm = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    
    #Loss functions
    lss = ['sparse_categorical_crossentropy','mean_squared_error','mean_absolute_error','mean_absolute_percentage_error',
           'mean_squared_logarithmic_error','squared_hinge','hinge','categorical_hinge'
           'logcosh']
    
    opt = sgd
    print('Creating model....')
    model = keras.Sequential([
        #Convolutional layer; kernel size was 80
        keras.layers.Conv1D(filters=100, kernel_size=80,strides=1, padding='valid',
                            data_format='channels_last', dilation_rate=1, activation=None,
                            use_bias=True, kernel_initializer='glorot_uniform',
                            input_shape=(pix, pix)),
        
        #Flatten layer condenses arrays
        keras.layers.Flatten(),
        #Dropout layer
        keras.layers.Dropout(0.1),
        #Final layer predicts
        keras.layers.Dense(65, activation=tf.nn.softmax)
        ])
    
    
    #Compile model
    print('Compiling model...')
    model.compile(optimizer=sgd, 
              loss = lss[0],
              metrics=['accuracy'])
    return(model)
    
def get_data2(fTrain, fTest, trn_imRAs, tst_imRAs, trn_cls, tst_cls, tst_nms):
    #Get train data
    trainfiles = [f for f in listdir(fTrain) if isfile(join(fTrain, f))]
    testfiles = [f for f in listdir(fTest) if isfile(join(fTest, f))]
    for f in trainfiles:
        if f.startswith('.'):
            continue
            
        trn_img = Image.open(fTrain+f)
        a = np.array(trn_img)
        trn_imRAs.append(a)
        c = int(bact2strn(f))
        trn_cls.append(c)
    cnt = 0   
    #Get test data
    for f in testfiles:
        if f.startswith('.'):
            continue
#        print('I\'m really trying....')
        tst_nms.append(f)
        tst_img = Image.open(fTest+f)
        a = np.array(tst_img)
        tst_imRAs.append(a)
        c = int(bact2strn(f))
        tst_cls.append(c)
        cnt+=1
        if (cnt==4):
            print('Send help')
    
def lbl_to_cls(lbl):
    strains = [1, 2, 3, 5, 9, 14, 23, 26, 29, 32, 33, 38, 41, 45, 47, 50, 52, 69, 73,
    78, 84, 86, 90, 94, 109, 113, 120, 122, 124, 125, 126, 127, 128, 129,
    132, 133, 134, 148, 149, 158, 159, 169, 174, 175, 210, 226, 230, 231,
    232, 237, 259, 276, 293, 298, 304, 313, 317, 329, 331, 333, 334, 337,
    338, 340, 344]

    outRA = []
    for i in range(len(lbl)):
        outRA.append(strains[i])
    return(outRA)


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
    
