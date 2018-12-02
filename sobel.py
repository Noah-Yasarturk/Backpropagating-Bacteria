#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:33:02 2018

@author: jeffreybruggeman
"""
import cv2
import numpy as np
from scipy import stats, ndimage, misc
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
#print("test")

def main():
    print("test")
    folder="/Users/jeffreybruggeman/git/Machine-Learning/Bacteria Project/Bacteria Dataset/Generated/"
    folderIn = f'{folder}/Circle/'
    folderOut = f'{folder}/Sobel/'
    print("test")
    sobel(folderIn, folderOut)
    return(0)
    
def bact_img(strain, copy_number):
    fileString = ('PIL-'+str(strain)+'_3dayLBCR-'+str(copy_number)+'.jpg')
    return(fileString)  
    
def sobel(folderIn_path, folderOut_path):
    print("test")
    for strn_num in range(1,345):
        for strn_copy in range(1,5):
            
            folderPath = folderIn_path
            folder_out = folderOut_path
            
            thisImg_Path_In = (folderPath + bact_img(str(strn_num),str(strn_copy)))
            thisImg_Path_Out = (folder_out + bact_img(str(strn_num),str(strn_copy)))
            thisImg_alone = bact_img(str(strn_num),str(strn_copy))

            if not (os.path.isfile(thisImg_Path_In)):
                continue
            print("test")
            thisImg = Image.open(thisImg_Path_In)
            
            sobelArr = ndimage.sobel(thisImg, mode='constant')
            
            sobelImg = Image.fromarray(sobelArr)
            
            sobelImg.save(thisImg_Path_Out, optimize=True, quality=95)
            
if __name__ == "__main__":
    main()
        
    
# =============================================================================
# folder="/Users/jeffreybruggeman/git/Machine-Learning/Bacteria Project/Bacteria Dataset/Generated/"
# image1=f"{folder}Circle/PIL-304_3dayLBCR-1.jpg"
# image2=f"{folder}Circle/PIL-304_3dayLBCR-2.jpg"
# image3=f"{folder}Circle/PIL-304_3dayLBCR-3.jpg"
# 
# fig = plt.figure()
# plt.gray()
# ax1=fig.add_subplot(321)
# ax2=fig.add_subplot(322)
# ax3=fig.add_subplot(323)
# ax4=fig.add_subplot(324)
# ax5=fig.add_subplot(325)
# ax6=fig.add_subplot(326)
# bact1=Image.open(image1)
# bact2=Image.open(image2)
# bact3=Image.open(image3)
# trans1=ndimage.sobel(bact1, mode='constant')
# trans2=ndimage.sobel(bact2, mode='constant')
# trans3=ndimage.sobel(bact3, mode='constant')
# ax1.imshow(bact1)
# ax2.imshow(trans1)
# ax3.imshow(bact2)
# ax4.imshow(trans2)
# ax5.imshow(bact3)
# ax6.imshow(trans3)
# plt.show()
# =============================================================================


