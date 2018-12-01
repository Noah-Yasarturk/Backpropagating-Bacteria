from PIL import Image, ImageChops
import numpy as np
import cv2

#HORIZONTAL METHODS######################################################

#main horizontal method: returns modified image.
def cleanGridLinesHoriz(fn):
    # main fn: we input an image file, get size and store dimensions separately
    img = cv2.imread(fn)
    height = np.size(img, 0)
    width = np.size(img, 1)

    #this is for future use, we will collect the height indices of the gridline rows, and use this
    #to provide our erasing method the exact coordinates of the gridlines it needs to erase :)
    gridLineFoundRowIndex = []

    #set a counter var that will loop from 1 to y
    heightLooper = 0

    #set a boolean var to hold the results of a future method
    gridLineCheckFlag = None
    #we scan the leftmost pixels of an image

    #this will check each row to determine if it is a gridline (TRUE) or not a gridline (FALSE)
    #if the row is a gridline (TRUE) then we add it to an array which we will use later to properly erase the lines
    while heightLooper < height:

        #call function with arg of heightLooper and image
        gridLineCheckFlag = detectGridLinesHoriz(heightLooper, img, width)

        #this will add the detected gridline to our 100% confirmed gridline-catching cache
        if gridLineCheckFlag == True:
            gridLineFoundRowIndex.append(heightLooper)

        #standard loop increment
        heightLooper = heightLooper + 1

    #we will use the cache to bring up the specific rows that are confirmed gridlines, and iterate through
    #the cache array to feed each gridline into the erase function
    for g in range(len(gridLineFoundRowIndex)):
        img[1:,g] = eraseGridLinesHoriz(g, img, width)

    return img

#helper method #1: returns boolean upon investigating a row
def detectGridLinesHoriz(HL, img, w):
    #the current row's height value, the image, the width of the image are all saved for local / temp. use
    currentHeight = HL
    im = img
    width = w

    #this is a variable that will determine if the row's pixels are equal to each other, thus forming a gridline
    rowFlag = None

    #standard loop increment variable, except this one we're moving from left to right through the row
    widthLooper = 0

    #this is the base case, will test the first two pixels on the left hand side of the row to see if they are equal
    if im[widthLooper,currentHeight] == im[widthLooper+1,currentHeight]:
        rowFlag = True
    else
        rowFlag = False

    #this is going to iterate through the rest of the row and compare each pixel to the one to its immediate left
    while widthLooper < width:
        if im[widthLooper, currentHeight] == im[widthLooper+1, currentHeight]
            rowFlag = rowFlag and True  #this updates our row checking variable
        else
            rowFlag = rowFlag and False #if a single value is out of place, this will not flag the row as a gridline

        #standard loop increment
        widthLooper = widthLooper + 1

    #this function will return whether it detected a complete gridline or not
    return rowFlag

#helper method #2: returns a modified row from the image itself
def eraseGridLinesHoriz(HL, img, w):
    # the current row's height value, the image, the width of the image are all saved for local / temp. use
    currentHeight = HL
    im = img
    width = w

    # standard loop increment variable, except this one we're moving from left to right through the row
    widthLooper = 0

    # this is going to iterate through the row, erasing the gridline by blending the pixel color as an average
    # of the colors of the pixel above and the pixel below it
    while widthLooper < width:

        #this is the blending / erasing magic
        if currentHeight == 0:
            #in the unlikely case the gridline is the top border of the image for some ungodly reason
            im[widthLooper, currentHeight] = im[widthLooper, currentHeight + 1]
        if currentHeight == np.size(im, 0):
            #in the unlikely case the gridline is the bottom border of the image for another ungodly reason
            im[widthLooper, currentHeight] = im[widthLooper, currentHeight - 1]
        else:
            #for the 99.8% of cases where the gridline is a normal, healthy vertical gridline that isn't a border
            im[widthLooper, currentHeight] = (im[widthLooper, currentHeight + 1] + im[widthLooper, currentHeight - 1]) / 2

        # standard loop increment
        heightLooper = heightLooper + 1;



    #this will return the erased / blended gridline, or what was a gridline
    return img[1:, currentHeight]


#VERTICAL METHODS########################################################

#main vertical method: returns modified image
def cleanGridLinesVert(fn):
    # main fn: we input an image file, get size and store dimensions separately
    img = cv2.imread(fn)
    height = np.size(img, 0)
    width = np.size(img, 1)

    # this is for future use, we will collect the weight indices of the gridline columns, and use this
    # to provide our erasing method the exact coordinates of the gridlines columns it needs to erase :)
    gridLineFoundColIndex = []

    # set a counter var that will loop from 1 to x
    widthLooper = 0

    # set a boolean var to hold the results of a future method
    gridLineCheckFlag = None
    # we scan the leftmost pixels of an image

    # this will check each column to determine if it is a gridline (TRUE) or not a gridline (FALSE)
    # if the row is a gridline (TRUE) then we add it to an array which we will use later to properly erase the lines
    while widthLooper < weight:

        # call function with arg of weightLooper and image
        gridLineCheckFlag = detectGridLinesHoriz(widthLooper, img, height)

        # this will add the detected gridline to our 100% confirmed gridline-catching cache
        if gridLineCheckFlag == True:
            gridLineFoundColIndex.append(weightLooper)

        # standard loop increment
        weightLooper = weightLooper + 1

    # we will use the cache to bring up the specific rows that are confirmed gridlines, and iterate through
    # the cache array to feed each gridline into the erase function
    for g in range(len(gridLineFoundColIndex)):
        img[g, 1:] = eraseGridLinesCol(g, img, height)

    return img

#helper method #1: returns boolean upon investigating a column
def detectGridLinesVert(WL, img, h):
    #the current column's width value, the image, the height of the image are all saved for local / temp. use
    currentWidth = WL
    im = img
    height = h

    #this is a variable that will determine if the columns's pixels are equal to each other, thus forming a gridline
    colFlag = None

    #standard loop increment variable, except this one we're moving from top to bottom through the column
    heightLooper = 0

    #this is the base case, will test the first two pixels on the top side of the column to see if they are equal
    if im[currentWidth,heightLooper] == im[currentWidth,heightLooper + 1]:
        colFlag = True
    else
        colFlag = False

    #this is going to iterate through the rest of the column and compare each pixel to the one immediately below it
    while heightLooper < height:
        if im[currentWidth, heightLooper] == im[currentWidth, heightLooper + 1]
            colFlag = colFlag and True  #this updates our row checking variable
        else
            colFlag = colFlag and False #if a single value is out of place, this will not flag the row as a gridline

        #standard loop increment
        heightLooper = heightLooper + 1

    #this function will return whether it detected a complete gridline or not
    return colFlag

#helper method #2: returns a modified column from the image itself
def eraseGridLinesVert(WL, img, h):
    # the current row's width value, the image, the height of the image are all saved for local / temp. use
    currentWidth = WL
    im = img
    height = h

    # standard loop increment variable, except this one we're moving from top to bottom through the column
    heightLooper = 0

    # this is going to iterate through the column, erasing the gridline by blending the pixel color as an average
    # of the colors of the pixel to its left and the pixel to its right
    while heightLooper < height:

        #this is the blending / erasing magic
        if currentWidth == 0:
            #in the unlikely case the gridline is the left border of the image for some ungodly reason
            im[currentWidth, heightLooper] = im[currentWidth + 1, heightLooper]
        if currentWidth == np.size(im, 1):
            #in the unlikely case the gridline is the right border of the image for another ungodly reason
            im[currentWidth, heightLooper] = im[currentWidth - 1, heightLooper]
        else:
            #for the 99.8% of cases where the gridline is a normal, healthy vertical gridline that isn't a border
            im[currentWidth, heightLooper] = (im[currentWidth + 1, heightLooper] + im[currentWidth - 1, heightLooper]) / 2

        # standard loop increment
        heightLooper = heightLooper + 1

    #this will return the erased / blended gridline, or what was a gridline
    return img[currentWidth, 1:]

#

#GAMEPLAN:
#create 2 copies of the image
#run cleanGridLinesHoriz() on the first image
#run cleanGridLinesVert() on the second image
#https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil
#we will average the two pictures to create an image without gridlines