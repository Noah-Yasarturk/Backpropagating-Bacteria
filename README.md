# Backpropagating-Bacteria
This project is conducted during Pimental-Alarcon's Fall 2018 Intro to ML course using Pseudomonas aeruginosa colony culture images for the purpose of developing a model that might be able to classify strains of the bacteria.

The code is specific to the image files we were given; the images are not to be distributed to the general public for the sake of protecting costly research. The images consisted of 2-4 images each of 65 strains of bacteria totalling 266 images.

Program Flow:


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


