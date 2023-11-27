
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io

def binary(input_image, threshold):
    #Create output array, start all black
    out = np.zeros_like(input_image)
    
    #White pixels where input image pixel > threshold
    out[input_image > threshold] = 1
    
    return out

def load(img_path):
    #Convert image path into nupty array 'image'
    image = np.asarray(io.imread(img_path))
    
    #Convert all pixel values to a range between 0.0 and 1.0
    range = np.array([1/255])
    
    #Sets output 'out' to the array of the shape
    out = range*image
    
    return out


def print_stats(image):
    #Set variables
    image_height, image_width, n_channels = image.shape
    
    #Print the stats
    print("Image height: ", image_height)
    print("Image width: ", image_width)
    print("Number of channels: ", n_channels)
    
    return None

def crop(image, start_row, start_col, num_rows, num_cols):
    #Slice from start to start+num required, etc
    out = image[start_row:start_row+num_rows, start_col:start_col+num_cols]
    
    return out


def change_contrast(image, factor):
    #Since x_n is essentially the output, and x_p is the input image, this only needs one line
    out = factor * (image - 0.5) + 0.5
    
    return out


def resize(input_image, output_rows, output_cols):
    #Create output array with output row and columns
    out = np.zeros((output_rows, output_cols, 3))
    
    #Find the ratio of the input/output rows and columns
    input_rows, input_cols, _ = input_image.shape
    rscale = input_rows/output_rows
    cscale = input_cols/output_cols
    
    #Nearest neighbor method
    for i in range(output_rows):
        for j in range(output_cols):
            out[i,j] = input_image[int(i*rscale), int(j*cscale)]

    return out


def greyscale(input_image):
    #Numpty mean function does this. Axis=2 points to channel, to average the 3 colours.
    out = np.mean(input_image, axis=-1, keepdims=True)

    return out


def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    
    """
    #Set Hi, Wi, Hk, Wk, create output 
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    #Find kernel centre, floor because odd
    ci = Hk // 2
    cj = Wk // 2
    
    #Nested loops to convolve
    for i in range(Hi):
        for j in range(Wi):
            #sum of convolution
            csum = 0
            for k in range(Hk):
                for l in range(Wk):
                    x = i-ci+k
                    y = j-cj+l
                    if x>=0 and x<Hi and y>=0 and y<Wi:
                        csum += image[x,y]*kernel[k,l]
            out[Hi-1-i, Wi-1-j] = csum
            
    return out

def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    print("expected_output\n", expected_output)
    print("test_output\n", test_output)
    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    #3 channels
    if len(image.shape) == 3:
        image_height, image_width, image_channels = image.shape
        out = np.zeros((image_height, image_width, image_channels))
        #applying the 2D convolution to each channel independently
        for image_channels in range(image_channels):
            out[:, :, image_channels] = conv2D(image[:, :, image_channels], kernel)
    #2 channels, pass directly into conv2D
    elif len(image.shape) == 2:
        out = conv2D(image, kernel)

    out = np.fliplr(np.flipud(out))
    
    return out

    
def gauss2D(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()