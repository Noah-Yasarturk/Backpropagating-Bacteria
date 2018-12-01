# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 07:23:40 2018

@author: nyasa
"""

'''
This program will convert the images to greyscale, resize them to 750 pixels, and crop the images.
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, ndimage


def main():
    folderIn = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/Dataset/PIL_3dayLBCR-training/'
    folderOut = 'C:/Users/nyasa/Documents/Classes/IntroML/Project/Reformatted Images 3/'
    #Suppress DecompressionBombError
    Image.MAX_IMAGE_PIXELS = None
    
    resizeAll_nSave(folderIn,folderOut,750,1,1)
    
    
    
    
    return(0)
    
def resizeAll_nSave(folderIn_path, folderOut_path, newPix_size, gray, CoM):
    '''
    This resizes all images to the newPix_size and makes them grayscale if gray==1.
    If CoM==1, it also trims based on center of mass (thank you, Jeff).
    '''
    
    #Paths to image
    folderPath = folderIn_path
    folder_out = folderOut_path
    
    #Create list to store aspect ratios
    asp_rats = []
    #Create list to store trimmed image widths
    widths = []
    #Create list to store image arrays
    img_RAs = []
    #Create list to store image names
    img_nms = []
    
    #blank variable
    nothing = 0
    
    #Iterate over our images
    print('Opening input images...')
    
    for strn_num in range(1,345):
        for strn_copy in range(1,5):
            #Handle hidious strain 13
            if (strn_num == 13):
                nothing +=1
                print('Skipping crappy strain 13 image')
            else:             
                try:
                    thisImg_Path_In = (folderPath + bact_img(str(strn_num),str(strn_copy)))
                    thisImg_Path_Out = (folder_out + bact_img(str(strn_num),str(strn_copy)))
                    thisImg_alone = bact_img(str(strn_num),str(strn_copy))
                    if (gray==0):
                        thisImg = Image.open(thisImg_Path_In)
                    if (gray==1):
                        thisImg = Image.open(thisImg_Path_In).convert('L')
                    thisImg_Wd, thisImg_Ht = thisImg.size
                    #Find aspect ratio
                    thisImg_aspRatio = (thisImg_Wd/thisImg_Ht)
                                  
                    print('Converting strain '+str(strn_num)+', copy '+str(strn_copy)+' to resized pixels.')
                    #Create resized image
                    ##min_img_wd has minumum pixel width; find proper height for this image using 
                    ##this image's aspect ratio
                    new_ht = (newPix_size/thisImg_aspRatio)
                    thisImg2 = thisImg.resize((newPix_size,int(new_ht)), Image.ANTIALIAS)
                    if (CoM==1):
                        #Convert to array
                        a = np.array(thisImg2)
                        #Save second copy for trimming
                        arr_1 = np.array(thisImg2)
                        #Find dimensions of original for trimming
                        dim_orig = np.array(thisImg2).shape
                        #Find mode of first x and y lines to establish background noise
                        mode_1 = stats.mode(a[0])
                        mode_2 = stats.mode(a[:,0])
                        #Take min of those modes for ideal threshold
                        mode_min = min(mode_1[0], mode_2[0])
                        #Set all pixels 30 under threshold and
                        a[a>mode_min-20] = 0
                        #check center of mass 1st time
                        CoM1 = ndimage.measurements.center_of_mass(a)
                        #Trim all 0 rows and columns
                        a = a[~np.all(a==0, axis=1)]
                        a=a[:,~np.all(a==0, axis=0)]
                        #Check center of mass second time
                        CoM2 = ndimage.measurements.center_of_mass(a)
                        #find dimensions of a for trimming
                        dim_a=a.shape
                        #Find image shift based on center of mass
                        dimension1=CoM1[0]-CoM2[0]
                        dimension2=CoM1[1]-CoM2[1]
                        #trim original image
                        arr_2=arr_1[int(dimension1):int(dim_a[0]+dimension1), int(dimension2):int(dim_a[1]+dimension2)]
                        #Create temp image
                        thisImg3 = Image.fromarray(arr_2)
                        #Add to list of image arrays
                        img_RAs.append(arr_2)
                        #Add aspect ratio to list of aspect ratios
                        thisImg3_Wd, thisImg3_Ht = thisImg3.size
                        thisImg3_aspRatio = (thisImg3_Wd/thisImg3_Ht)
                        asp_rats.append(thisImg3_aspRatio)
                        widths.append(thisImg3_Wd)
                        #Add image name to list
                        img_nms.append(thisImg_Path_Out)
                                          
                    else:
                        #We aren't considering Center of Mass
                        thisImg2.save((folder_out + thisImg_alone), optimize=True, quality=95)
                        
                except IOError:
                    nothing +=1
    if (CoM == 1):
        print('')
        print('Done converting images to arrays and finding aspect ratios.')
        print('Now begin reconstruction of images with universalized dimensions.')
        #Find minimum width
        min_wd = min(widths)
        for i in range(len(img_RAs)):
            #Create image from the ith image array
            thisImg4 = Image.fromarray(img_RAs[i])
            #Resize it to the appropriate dimensions
            new_ht = (min_wd/asp_rats[i])
            thisImg4 = thisImg4.resize((min_wd,int(new_ht)), Image.ANTIALIAS)
            #Save it
            print('Saving image '+img_nms[i])
            thisImg4.save((img_nms[i]), optimize=True, quality=95)
    
    
def bact_img(strain, copy_number):
    fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number)+'.jpg')
    return(fileString)  
    

if __name__ == "__main__":
    main()