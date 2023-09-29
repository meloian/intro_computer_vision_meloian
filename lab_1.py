# Introduction to computer vision, lab_1, FI-21 Meloian Myroslav, 15

import cv2
import numpy as np

# convert the image to a b/w version
def convert_to_bw(image):
    bw_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_color = image[i, j]
            gray_value = int(0.299 * pixel_color[2] + 0.587 * pixel_color[1] + 0.114 * pixel_color[0])
            bw_image[i, j] = gray_value
    return bw_image

# binarize the image and get a mask
def binarize(image, threshold=128):
    binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            if pixel_value < threshold:
                binary_mask[i, j] = 255  # make the object white
            else:
                binary_mask[i, j] = 0    # make the background black
    return binary_mask

# cut out the object using a mask
def cutout_object(original_image, binary_mask):
    cutout_image = np.zeros_like(original_image)
    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            if binary_mask[i, j] == 255:
                cutout_image[i, j] = original_image[i, j]
    return cutout_image

# process the image using the above functions
def process_image(my_input_image, output_prefix, border_value):
    input_image = cv2.imread(my_input_image)
    bw_image = convert_to_bw(input_image)
    binary_mask = binarize(bw_image, border_value)
    cutout_image = cutout_object(input_image, binary_mask)

    # replace the background with black
    cutout_image[np.where((cutout_image == [0, 0, 0]).all(axis=2))] = [0, 0, 0]

    # get the image number from the file name
    image_number = my_input_image.split('_')[-1].split('.')[0]

    # save the processed images with appropriate names
    cv2.imwrite(output_prefix + '_bw_' + image_number + '.jpg', bw_image)
    cv2.imwrite(output_prefix + '_binary_mask_' + image_number + '.jpg', binary_mask)
    cv2.imwrite(output_prefix + '_cutout_object_' + image_number + '.jpg', cutout_image)

# input data
my_input_image = 'my_image1.jpg'
output_prefix = 'output'
border_value = 230

# process and save the image
process_image(my_input_image, output_prefix, border_value)


