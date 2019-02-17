# Backpropagating-Bacteria
This project is conducted during Pimental-Alarcon's Fall 2018 Intro to ML course using Pseudomonas aeruginosa colony culture images for the purpose of developing a model that might be able to classify strains of the bacteria.

The code is specific to the image files we were given; the images are not to be distributed to the general public for the sake of protecting costly research. The images consisted of 2-4 images each of 65 strains of bacteria totalling 266 images.

Python Dependencies:
Ensure all of the following programs are installed and updated.
1. tensorflow
2. numpy
3. Python Image Library (PIL)
4. matplotlib
5. scipy

Program Flow:
1. Ensure image dataset is in same directory as Python executables.
2. Run convert_imgs.py to perform the below methods and create our training and testing sets (80% and 20% of data, respectively).
3. Run make_NN_final.py and wait 20-45 minutes for model generation. Evaluation will be stored in same directory as "predictions.csv".


METHODS
Image Standardization:
Images were universalized to 100 x 100 pixels centered at center of mass to ensure consistent NumPy arrays for neural network input.

Feature Extraction/Data Cleaning:
To ensure that the neural network would effectively differentiate amongst the classes, we:
1. Used a combination of center-of-mass detection and applied a radius from CoM to exclude the noisy background of the colony culture
2. Generated Sobel images from the dataset to emphasize the edges of the colony

Image Mulitplication:
To overcome our limited dataset, we:
1. Converted the images to grayscale using multiple thresholds
2. Rotated the images at 15 degree intervals

RESULTS


