# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:29:48 2018

@author: nyasa
"""

'''
In this program,  I hope to implement tensorflow on our images.
'''

from shutil import copyfile
from pathlib import Path
import random
import sys

def main():
    #Separate testing and training data
    folderIn_path = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/Reformatted Images 3/'
    #Training set
    folderOut_path1 = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/training_images/'
    #Testing set
    folderOut_path2 = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/testing_images/'

    
    ##For strains with 2-4 images, randomly select 1 for testing
    print('Creating testing set')
    for strn_num in range(1,345):
        #Handle hidious strain 13
        if (strn_num == 13):
            print('Skipping crappy strain 13 image')
            
        else:
            file_Exists = False
            checkStrnFile = Path(folderIn_path+bact_img(strn_num,1))
            if (checkStrnFile.is_file()==True):
                while (file_Exists == False):
                    #Select a random number from 1-4 to select test image
                    rand_strn = random.randint(1,4)
                    thisImg_path = (folderIn_path + bact_img(strn_num, rand_strn))
                    my_file = Path(thisImg_path)
                    if (my_file.is_file() == False):
                        #The file doesn't exist; try again on a new file
                        file_Exists = False
                    else:
                        #The file exists; add to testing set
                        print('Copying '+bact_img(strn_num, rand_strn) +' to testing set.')
                        copyfile(thisImg_path, (folderOut_path2 + bact_img(strn_num, rand_strn)))
                        file_Exists = True
    ##For all other files, copy over to the training set
    print('Creating training set')
    for strn_num in range(1,345):
        #Iterate through strains with 4 copies
        checkStrnFile = Path(folderIn_path+bact_img(strn_num,4))
        if (checkStrnFile.is_file()==True):
            for strn_copy in range(1,5):
                #Make sure we don't copy from testing set
                check_inFile = Path(folderOut_path2 + bact_img(strn_num,strn_copy))
                if (check_inFile.is_file() == False):
                    thisImg = bact_img(strn_num, strn_copy)
                    thisFile = Path(folderOut_path2+thisImg)
                    #Check to see if it's in our testing set
                    if (thisFile.is_file() == False):
                        #Since it isn't, copy over the file from the original folder to the training set
                        print('Copying '+thisImg+' to training set.')
                        copyfile((folderIn_path+thisImg), (folderOut_path1+thisImg))
                        
        #Iterate through strains with 3 copies
        checkStrnFile3 = Path(folderIn_path+bact_img(strn_num,3))
        if (checkStrnFile3.is_file()==True) and (checkStrnFile.is_file()==False):
            for strn_copy in range(1,4):
                #Make sure we don't copy from testing set
                check_inFile = Path(folderOut_path2 + bact_img(strn_num,strn_copy))
                if (check_inFile.is_file() == False):
                    thisImg = bact_img(strn_num, strn_copy)
                    thisFile = Path(folderOut_path2+thisImg)
                    #Check to see if it's in our testing set
                    if (thisFile.is_file() == False):
                        #Since it isn't, copy over the file from the original folder to the training set
                        print('Copying '+thisImg+' to training set.')
                        copyfile((folderIn_path+thisImg), (folderOut_path1+thisImg))
                        
        #Iterate through strains with 2 copies
        checkStrnFile2 = Path(folderIn_path+bact_img(strn_num,2))
        if (checkStrnFile2.is_file()==True) and (checkStrnFile.is_file()==False) and (checkStrnFile3.is_file()==False):
            for strn_copy in range(1,3):
                #Make sure we don't copy from testing set
                check_inFile = Path(folderOut_path2 + bact_img(strn_num,strn_copy))
                if (check_inFile.is_file() == False):
                    thisImg = bact_img(strn_num, strn_copy)
                    thisFile = Path(folderOut_path2+thisImg)
                    #Check to see if it's in our testing set
                    if (thisFile.is_file() == False):
                        #Since it isn't, copy over the file from the original folder to the training set
                        print('Copying '+thisImg+' to training set.')
                        copyfile((folderIn_path+thisImg), (folderOut_path1+thisImg))
        print('Testing and training sets successfully randomly made.')
    
def bact_img(strain, copy_number):
    fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number)+'.jpg')
    return(fileString)     

if __name__ == "__main__":
    main()
