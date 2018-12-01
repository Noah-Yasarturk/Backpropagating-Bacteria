#import cv2, numpy
import numpy

#how to check if gridline is more than 1px in width:
#in the blending / erasing step of the Erase Methods, we will include in the conditionals
#a check to see whether the pixels immediately above/below or left/right of the gridline
#are the same value. If they are, we will call another method that will recursively keep
#on going in the direction of the suspected extra width, until the pixels above/below or
#left/right of it no longer have the same value.

#also needed: averaging both final images
#see: https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil
#after that, we will need to properly install the CV2 and PIL modules