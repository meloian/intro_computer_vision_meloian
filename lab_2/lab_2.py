import cv2
import numpy as np

# load an image
image = cv2.imread('my_image3.jpg', 0)  # read the image as grayscale (if needed)

# function to save an image
def save_image(output_image, filename):
    cv2.imwrite(filename, output_image)

# 1. shift the image 10 pixels to the right and 20 pixels down
def shift_image(image, dx, dy):
    height, width = image.shape
    shifted_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            # calculate new pixel coordinates
            new_x, new_y = x + dx, y + dy
            # check if new coordinates are within image bounds
            if 0 <= new_x < width and 0 <= new_y < height:
                shifted_image[new_y, new_x] = image[y, x]
    return shifted_image

translated_image = shift_image(image, 10, 20)
save_image(translated_image, 'translated_image.jpg')

# 2. color inversion
inverted_image = 255 - image
save_image(inverted_image, 'inverted_image.jpg')

# 3. gaussian blur
def gaussian_blur(image, kernel_size):
    blurred_image = np.copy(image)
    k = kernel_size // 2
    height, width = image.shape
    for y in range(k, height - k):
        for x in range(k, width - k):
            # calculate the mean value of pixels in a window as a blurring effect
            blurred_image[y, x] = np.mean(image[y-k:y+k+1, x-k:x+k+1])
    return blurred_image

blurred_image = gaussian_blur(image, 11)
save_image(blurred_image, 'blurred_image.jpg')

# 4. diagonal motion blur
def motion_blur(image, kernel_size):
    kernel = np.eye(kernel_size) / kernel_size
    motion_blur_image = cv2.filter2D(image, -1, kernel)
    return motion_blur_image

motion_blur_image = motion_blur(image, 7)
save_image(motion_blur_image, 'motion_blur_image.jpg')

# 5. function to apply the "Sharpening" filter
def custom_sharpen_image(image):
    height, width = image.shape
    sharpened_image = np.zeros((height, width), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = 10 * image[y, x]
            neighbors = image[y-1:y+2, x-1:x+2].flatten()
            # calculate the difference between the central pixel and its neighbors
            sharpened_pixel = center - neighbors.sum()
            sharpened_image[y, x] = np.clip(sharpened_pixel, 0, 255)  # Use np.clip to constrain values
    return sharpened_image

sharpened_image = custom_sharpen_image(image)
save_image(sharpened_image, 'sharpened_image.jpg')

# 6. function to apply the "Sobel Filter"
def custom_sobel_filter(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    height, width = image.shape
    sobel_image = np.zeros((height, width), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            pixel_region = image[y-1:y+2, x-1:x+2]
            gradient_x = (sobel_x * pixel_region).sum()
            gradient_y = (sobel_y * pixel_region).sum()
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            sobel_image[y, x] = np.clip(gradient_magnitude, 0, 255)
    return sobel_image

sobel_image = custom_sobel_filter(image)
save_image(sobel_image, 'sobel_image.jpg')

# 7. function to apply the "Edge Detection" filter
def custom_edge_detection(image, low_threshold, high_threshold):
    gradient_magnitude = custom_sobel_filter(image)
    height, width = image.shape
    edge_image = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            if gradient_magnitude[y, x] >= high_threshold:
                edge_image[y, x] = 255
            elif gradient_magnitude[y, x] >= low_threshold:
                edge_image[y, x] = 50  # pixels meeting the low threshold are set to 50 (gray)
    
    return edge_image

edge_image = custom_edge_detection(image, 100, 200)
save_image(edge_image, 'edge_image.jpg')

# 8. Function to apply the my filter
def my_filter(image, block_size):
    height, width = image.shape
    mosaic_image = np.copy(image)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            average_color = np.mean(block)
            mosaic_image[y:y+block_size, x:x+block_size] = average_color
    return mosaic_image

# Use this function to apply the my filter
mosaic_image = my_filter(image, 20)  # block size: 20x20
save_image(mosaic_image, 'my_filter_image.jpg')



