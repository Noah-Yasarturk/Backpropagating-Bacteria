# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:29:48 2018
@author: nyasa
"""

'''
This separates the training and testing sets, both Sobel and Circle.
'''

from shutil import copyfile
from pathlib import Path
import random
import sys

def main():
    #Separate testing and training data
    fIn_path = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/rs_set/'
    #Training set
    fOut_path1 = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/rs_set/training_images_rs/'
    #Testing set
    fOut_path2 = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/rs_set/testing_images_rs/'

    
    ##For strains with 2-4 images, randomly select 1 for testing
    print('Creating testing set')
    for strn_num in range(1,345):
        #Handle hidious strain 13
        if (strn_num == 13):
            print('Skipping crappy strain 13 image')
            
        else:
            file_Exists = False
            #Check if strain exists
            checkStrnFile_cir = Path(fIn_path+ 'all_cirs/'+bact_img(strn_num,1,0))
            checkStrnFile_Sbl = Path(fIn_path+ 'all_Sobel/'+bact_img(strn_num,1,0))
            if (checkStrnFile_cir.is_file()==True) and (checkStrnFile_Sbl.is_file()==True):
                #If it does, select a random integer strain copy
                while (file_Exists == False):
                    #Select a random number from 1-4 to select test images
                    rand_strn = random.randint(1,4)
                    thisImg_path_cir = (fIn_path + 'all_cirs/'+bact_img(strn_num, rand_strn,0))
                    my_file_cir = Path(thisImg_path_cir)
                    thisImg_path_Sbl = (fIn_path + 'all_Sobel/'+bact_img(strn_num, rand_strn,0))
                    my_file_Sbl = Path(thisImg_path_Sbl)
                    if (my_file_cir.is_file() == False) or (my_file_Sbl.is_file() == False):
                        #The file doesn't exist; try again on a new file
                        file_Exists = False
                    else:
                        #The file exists; add to testing set
                        print('Copying '+bact_img(strn_num, rand_strn,0) +' to testing set.')
                        copyfile(thisImg_path_cir, (fOut_path2 + bact_img(strn_num, rand_strn,1)))
                        copyfile(thisImg_path_Sbl, (fOut_path2 + bact_img(strn_num, rand_strn,2)))
                        file_Exists = True
                        
                        
    ##For all other files, copy over to the training set
    print('Creating training set')
    for strn_num in range(1,345):
        #Iterate through strains with 4 copies
        checkStrnFile_cir = Path(fIn_path+ 'all_cirs/'+bact_img(strn_num,4,0))
        checkStrnFile_Sbl = Path(fIn_path+ 'all_Sobel/'+bact_img(strn_num,4,0))
        if (checkStrnFile_cir.is_file()==True) and (checkStrnFile_Sbl.is_file()==True):
            for strn_copy in range(1,5):
                #Make sure we don't copy from testing set
                check_inFile = Path(fOut_path2 + bact_img(strn_num,strn_copy,0))
                if (check_inFile.is_file() == False):
                    #If the image is not distributed to our test set already,
                    thisImg = bact_img(strn_num, strn_copy,0)
                    thisImg_c = bact_img(strn_num, strn_copy,1)
                    thisImg_s = bact_img(strn_num, strn_copy,2)
                    thisFile_cir = Path(fOut_path2+thisImg_c)
                    thisFile_Sbl = Path(fOut_path2+thisImg_s)
                    #Check to see if it's in our testing set
                    if (thisFile_cir.is_file() == False) and (thisFile_Sbl.is_file() == False):
                        #Since it isn't, copy over the file from the original folder to the training set
                        print('Copying to training set.')
                        copyfile((fIn_path+'all_cirs/'+thisImg), (fOut_path1+thisImg_c))
                        copyfile((fIn_path+'all_Sobel/'+thisImg), (fOut_path1+thisImg_s))
                        
                        
        #Iterate through strains with 3 copies
        checkStrnFile_cir = Path(fIn_path+ 'all_cirs/'+bact_img(strn_num,3,0))
        checkStrnFile_Sbl = Path(fIn_path+ 'all_Sobel/'+bact_img(strn_num,3,0))
        if (checkStrnFile_cir.is_file()==True) and (checkStrnFile_Sbl.is_file()==True):
            for strn_copy in range(1,4):
                #Make sure we don't copy from testing set
                check_inFile = Path(fOut_path2 + bact_img(strn_num,strn_copy,0))
                if (check_inFile.is_file() == False):
                    #If the image is not distributed to our test set already,
                    thisImg = bact_img(strn_num, strn_copy,0)
                    thisImg_c = bact_img(strn_num, strn_copy,1)
                    thisImg_s = bact_img(strn_num, strn_copy,2)
                    thisFile_cir = Path(fOut_path2+thisImg_c)
                    thisFile_Sbl = Path(fOut_path2+thisImg_s)
                    #Check to see if it's in our testing set
                    if (thisFile_cir.is_file() == False) and (thisFile_Sbl.is_file() == False):
                        #Since it isn't, copy over the file from the original folder to the training set
                        print('Copying to training set.')
                        copyfile((fIn_path+'all_cirs/'+thisImg), (fOut_path1+thisImg_c))
                        copyfile((fIn_path+'all_Sobel/'+thisImg), (fOut_path1+thisImg_s))
                        
                        
        #Iterate through strains with 2 copies
        checkStrnFile_cir = Path(fIn_path+ 'all_cirs/'+bact_img(strn_num,2,0))
        checkStrnFile_Sbl = Path(fIn_path+ 'all_Sobel/'+bact_img(strn_num,2,0))
        if (checkStrnFile_cir.is_file()==True) and (checkStrnFile_Sbl.is_file()==True):
            for strn_copy in range(1,3):
                #Make sure we don't copy from testing set
                check_inFile = Path(fOut_path2 + bact_img(strn_num,strn_copy,0))
                if (check_inFile.is_file() == False):
                    #If the image is not distributed to our test set already,
                    thisImg = bact_img(strn_num, strn_copy,0)
                    thisImg_c = bact_img(strn_num, strn_copy,1)
                    thisImg_s = bact_img(strn_num, strn_copy,2)
                    thisFile_cir = Path(fOut_path2+thisImg_c)
                    thisFile_Sbl = Path(fOut_path2+thisImg_s)
                    #Check to see if it's in our testing set
                    if (thisFile_cir.is_file() == False) and (thisFile_Sbl.is_file() == False):
                        #Since it isn't, copy over the file from the original folder to the training set
                        print('Copying to training set.')
                        copyfile((fIn_path+'all_cirs/'+thisImg), (fOut_path1+thisImg_c))
                        copyfile((fIn_path+'all_Sobel/'+thisImg), (fOut_path1+thisImg_s))
                        
                        
        print('Testing and training sets successfully randomly made.')
    
def bact_img(strain, copy_number, v):
    #v=0 is nothing, 1 is circle, 2 is Sobel
    if (v==0):
        fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number)+'.jpg')
    if (v==1):
        fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number)+'-c.jpg')
    if (v==2):
        fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number)+'-s.jpg')
    return(fileString)     

if __name__ == "__main__":
    main()
