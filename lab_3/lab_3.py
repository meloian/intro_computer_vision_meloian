import cv2
import numpy as np
import os

def erosion(image, kernel, iterations=1):
    # initialize the result image
    result = np.zeros_like(image)

    # perform erosion for the specified number of iterations
    for _ in range(iterations):
        # iterate through each pixel in the image
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                # check if the surrounding pixels match the kernel
                match = True
                for m in range(kernel.shape[0]):
                    for n in range(kernel.shape[1]):
                        if kernel[m, n] == 1 and image[i - 1 + m, j - 1 + n] != 255:
                            match = False
                            break
                # set the result pixel value based on the matching condition
                result[i, j] = 255 if match else 0

    return result

def dilation(image, kernel):
    # initialize the result image
    result = np.zeros_like(image)

    # iterate through each pixel in the image
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # check if any of the surrounding pixels are white based on the kernel
            dilate_value = False
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    if kernel[m, n] == 1 and image[i - 1 + m, j - 1 + n] == 255:
                        dilate_value = True
                        break
            # set the result pixel value based on the dilation condition
            result[i, j] = 255 if dilate_value else 0

    return result

def opening(image, kernel):
    # perform erosion followed by dilation to achieve opening
    eroded = erosion(image, kernel)
    result = dilation(eroded, kernel)

    return result

def closing(image, kernel):
    # perform dilation followed by erosion to achieve closing
    dilated = dilation(image, kernel)
    result = np.zeros_like(image, dtype=np.uint8)

    # iterate through each pixel in the image
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # compute the sum of pixel values in the kernel neighborhood after dilation
            sum_neighbors = np.sum(dilated[i - 1:i + 2, j - 1:j + 2] * kernel)
            # set the result pixel value based on the sum condition
            result[i, j] = 255 if sum_neighbors > 0 else 0

    return result

def gradient_magnitude(image):
    # define Sobel kernels for gradient computation in x and y directions
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # donvolve the image with Sobel kernels to compute gradients in x and y directions
    gradient_x = convolve(image, kernel_x)
    gradient_y = convolve(image, kernel_y)

    # compute the magnitude of the gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # normalize the magnitude to the range [0, 255]
    magnitude = (magnitude / np.max(magnitude)) * 255

    return magnitude.astype(np.uint8)

def convolve(image, kernel):
    # use OpenCV's filter2D function for convolution
    return cv2.filter2D(image, cv2.CV_64F, kernel)

def find_edges(image):
    # convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        image = np.mean(image, axis=-1).astype(np.uint8)

    # compute the gradient magnitude of the image
    edges = gradient_magnitude(image)

    return edges

# load the input image
image_path = 'my_image2.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# create a directory to save the results
output_directory = 'output_images'
os.makedirs(output_directory, exist_ok=True)

# define the kernel for morphological operations
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], np.uint8)

# apply morphological operations and edge detection
eroded_image = (erosion(original_image, kernel) > 0).astype(np.uint8) * 255
dilated_image = (dilation(original_image, kernel) > 0).astype(np.uint8) * 255
opened_image = (opening(original_image, kernel) > 0).astype(np.uint8) * 255
closed_image = (closing(original_image, kernel) > 0).astype(np.uint8) * 255
edges_image = find_edges(original_image)

# save the results in the output directory
cv2.imwrite(os.path.join(output_directory, 'original_image.png'), original_image)
cv2.imwrite(os.path.join(output_directory, 'eroded_image.png'), eroded_image)
cv2.imwrite(os.path.join(output_directory, 'dilated_image.png'), dilated_image)
cv2.imwrite(os.path.join(output_directory, 'opened_image.png'), opened_image)
cv2.imwrite(os.path.join(output_directory, 'closed_image.png'), closed_image)
cv2.imwrite(os.path.join(output_directory, 'edges_image.png'), edges_image)

print("Results saved in the 'output_images' directory.")


