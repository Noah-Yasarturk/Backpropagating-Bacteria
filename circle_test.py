# =============================================================================
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Dec  1 14:33:27 2018
# 
# @author: jeffreybruggeman
# """
# 
# import cv2
# import matplotlib.pyplot as plt
# from scipy import stats
# #import cv2.cv as cv
# img1 = cv2.imread('Bacteria Dataset/Generated/PIL-304_3dayLBCR-4.jpg')
# img = cv2.imread('Bacteria Dataset/Generated/PIL-304_3dayLBCR-4.jpg',0)
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# mode=(stats.mode(img, axis=None))
# ret, thresh = cv2.threshold(gray, mode[0], 255, cv2.THRESH_BINARY)
# 
# plt.imshow(thresh)
# plt.show()
# edges = cv2.Canny(thresh, 50, 200)
# #cv2.imshow('detected ',gray)
# cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                                                          #175
# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 10, param2 = 10, minRadius = 220 , maxRadius = 250)
# for i in circles[0,:]:
#     i[2]=i[2]+4
#     cv2.circle(img1,(i[0],i[1]),i[2],(0,255,0),2)
# 
# #Code to close Window
# cv2.imshow('detected Edge 1',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 
# =============================================================================

# Based on code from https://stackoverflow.com/questions/36911877/cropping-circle-from-image-using-opencv-python?rq=1
# by users João Cartucho and Rahul Chougule
import cv2
import numpy as np
from scipy import stats
from PIL import Image
import pandas as pd
#import matplotlib.pyplot as plt
import os



def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

#file='PIL-26_3dayLBCR-1.jpg'
#file='PIL-41_3dayLBCR-2.jpg'
#file='PIL-9_3dayLBCR-2.jpg'
folder='/Users/jeffreybruggeman/git/Machine-Learning/Bacteria Project'
df2=pd.DataFrame(columns=['strain', 'image', 'mode', 'brightness'])
df = pd.read_excel(f"{folder}/Perron_phenotype-GSU-training.xlsx", sheet_name="Isolates w Microscopy")
col = df['strain']
for (k, Series) in col.iteritems():
#     img = cv2.imread("Bacteria Dataset"/PIL-"+strain+"_3dayLBCR-"+file+".jpg")
    for file in range(1,5):
        if not (os.path.isfile(f"{folder}/Bacteria Dataset/Generated/Square/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg")):
            continue
        img1 = cv2.imread(f"{folder}/Bacteria Dataset/Generated/Square/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg")
        img = cv2.imread(f"{folder}/Bacteria Dataset/Generated/Square/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg",0)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        mode=(stats.mode(img, axis=None))
        
        test_image=Image.open(f"{folder}/Bacteria Dataset/Generated/Square/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg")
#        min_thresh=calculate_brightness(test_image)*255*1.5
        #t = min(min_thresh, mode[0])
        #print(mode[0], " ", calculate_brightness(test_image))
        
        ########
#        print(mode[0], " ",calculate_brightness(test_image)," ", min_thresh)
        ########
        if(mode[0]==255):
            min_thresh=200
        else:
            min_thresh= (255*calculate_brightness(test_image))*1.3
#        print("min_thresh= ", min_thresh)
        ret, thresh = cv2.threshold(gray, min_thresh, 255, cv2.THRESH_BINARY)
        
#        plt.imshow(thresh)
#        plt.show
        
        # Create mask
        height,width = img.shape
        mask = np.zeros((height,width), np.uint8)
        
        edges = cv2.Canny(thresh, 100, 200)
        #cv2.imshow('detected ',gray)
        cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 10, param2 = 10, minRadius = 150, maxRadius = 300)
        for i in circles[0,:]:
            i[2]=i[2]+4
            # Draw on mask
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
        
        #print(crop[:,:,1])
#        plt.imshow(thresh)
#        plt.show
        print(f"Saving strain {df['strain'].at[k]} image {file}")
#        print(f"{df['strain'].at[k]} {file} {mode[0]} ", calculate_brightness(test_image), " ", min_thresh)
        this = Image.fromarray(crop)
        img = this.resize((365,365), Image.ANTIALIAS)
        img.save(f"{folder}/Bacteria Dataset/Generated/Circle/PIL-{df['strain'].at[k]}_3dayLBCR-{file}.jpg", optimize=True, quality=95)
        

#Code to close Window
#cv2.imshow('detected Edge',img1)
#cv2.imshow('Cropped Bacteria',crop) 
#cv2.waitKey(0)
#cv2.destroyAllWindows()

