#!/usr/bin/env python3
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
from scipy import stats, ndimage
import cv2
import pandas as pd
import os

def main():
    folder='/'
    
    folderIn = 'Bacteria Dataset/'
    folderOut = 'Generated/Square/'
    folderCircle='Generated/Circle100/'
    folderFinal=f'Generated/train/'    #all generated images go here
    
    
    
    #Suppress DecompressionBombError
    Image.MAX_IMAGE_PIXELS = None
    
    resizeAll_nSave(folderIn,folderOut,750,1,1)
    
    circle(folder, folderOut, folderCircle)
    
    final(folder, folderCircle, folderFinal)
    
    
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
                    thisImg_Path_In = (folderPath + bact_img(str(strn_num),str(strn_copy)))+".jpg"
                    thisImg_Path_Out = (folder_out + bact_img(str(strn_num),str(strn_copy)))+".jpg"
                    thisImg_alone = bact_img(str(strn_num),str(strn_copy))+'.jpg'
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
#                        dim_orig = np.array(thisImg2).shape
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
        
        #######
        print(widths)
        #######
        
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

# Based on code from https://stackoverflow.com/questions/36911877/cropping-circle-from-image-using-opencv-python?rq=1
# by users João Cartucho and Rahul Chougule
def circle(folder, folderIn, folderCircle):
    # Load strain names from xlsx file    
    df = pd.read_excel("Perron_phenotype-GSU-training.xlsx", sheet_name="Isolates w Microscopy")
    col = df['strain']
    # iterate strain / images
    for (k, Series) in col.iteritems():
        for file in range(1,5):
            # avoid files that DNE and strain 331 image 4 as it generates poorly
            if not (os.path.isfile(f"Generated/Square/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg")):
                continue
            if (k==59 and file==4): # catches strain 331 image 4 as an outlier
                 continue
            # Loading for processing
            img1 = cv2.imread(f"Generated/Square/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg")
            img = cv2.imread(f"Generated/Square/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg",0)
            # Sets to gray
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            # finds mode to better ind threshold
            mode=(stats.mode(img, axis=None))
            
            test_image=Image.open(f"Generated/Square/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg")
            # This handles the darker vs lighter images so the hough circle gets more accurate
            if(mode[0]==255):
                min_thresh=200
            else:
                min_thresh= (255*calculate_brightness(test_image))*1.3
            # sets thresholds
            ret, thresh = cv2.threshold(gray, min_thresh, 255, cv2.THRESH_BINARY)
    
            # Create mask
            height,width = img.shape
            mask = np.zeros((height,width), np.uint8)
            # finds edges
            edges = cv2.Canny(thresh, 100, 200)
            # creates circles
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 10, param2 = 10, minRadius = 180, maxRadius = 230)
            for i in circles[0,:]:
                i[2]=i[2]+4
                # Draw on mask, was useful for testing before trimming
                cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
            
            # Copy that image using that mask
            masked_data = cv2.bitwise_and(img1, img1, mask=mask)
            
            # Apply Threshold
            _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
            
            # Find Contour
            contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv2.boundingRect(contours[0])
            
            # Crop masked_data
            crop = masked_data[y:y+h,x:x+w] # Crop is the one we want to save, Noah
            
            print(f"Saving strain {df['strain'].at[k]} image {file}")
            # crops 3d array to 2d, resizes to 100x100 saves depending on train / test use
            this = Image.fromarray(crop[:,:,0])
            img = this.resize((100,100), Image.ANTIALIAS)
            # Every file 3 is used in testing set
            if (file==3):
                img.save(f"Generated/test/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg", optimize=True, quality=95)
            else:
                img.save(f"Generated/Circle100/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg", optimize=True, quality=95)


def final(folder, folder_in, folder_out):
    # iterates again
    for strn_num in range(1,345):
            for strn_copy in range(1,5):
                # To avoid putting testing images into training set
                if (strn_copy==3):
                    continue
                # Sets paths
                thisImg_Path_In = (folder_in + bact_img(str(strn_num),str(strn_copy)))+".jpg"
                thisImg_Path_Out = (folder_out + bact_img(str(strn_num),str(strn_copy)))
                # Avoids images that DNE
                if not (os.path.isfile(thisImg_Path_In)):
                    continue
                # Degree to rotate for eachimage  rotation
                rotate_deg=15
                
                for rotate in range(0, int(360/rotate_deg)):
                    rotate_actual = rotate * rotate_deg
                    ## Start here
                    #Rotate image, save
                    thisImg = Image.open(thisImg_Path_In)
                    rot_img= thisImg.rotate(rotate_actual)
                    rot_img.save(thisImg_Path_Out+f"-c-r{rotate_actual}.jpg", optimize=True, quality=95)
                    # Sobel constant, save
                    sobelArr = ndimage.sobel(rot_img, mode='constant')
                    sobelImg = Image.fromarray(sobelArr)
                    sobelImg.save(thisImg_Path_Out+f"-c-r{rotate_actual}-s.jpg", optimize=True, quality=95)
                    # Threshold 1-3, save
                    i=1
                    # Different threshold modifiers
                    thr=[.9,1,1.1]
                    for k in thr:
                        # calc brightness returns a variable from 0-1 that I scale to 0-255
                        # then I multiply it by k to generate different thresholds
                        threshold =255*calculate_brightness(rot_img)*k
                        thresh = thisImg.point(lambda p: p > threshold and 255) 
                        thresh.save(thisImg_Path_Out+f"-c-r{rotate_actual}-t{i}.jpg", optimize=True, quality=95)
                        i=i+1

# Calc brightness helps to find relative brightness to help with thresholds
def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

# Noahs image name handler
def bact_img(strain, copy_number):
    fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number))
    return(fileString)  
    

if __name__ == "__main__":
    main()