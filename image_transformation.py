
# coding: utf-8

# # Image transformation functions
# 
# Here we have a collection of usefull functions to do image transformation

# In[1]:

import numpy as np
import cv2
import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt


# In[9]:

# display a set of images side by side
def plotImageSet(image_list,color='gnuplot2'):
    fig = plt.figure();
    fig, ax = plt.subplots(1, len(image_list), figsize=(24, 9));
    fig.tight_layout();
    count = 0
    if (len(image_list)>1):
        for image in image_list:
            ax[count].imshow(image,cmap=color);
            ax[count].axis('off');
            count = count +1
    else:
        ax.imshow(image_list[0],cmap=color);
        ax.axis('off');
    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.);
    plt.show();


# In[10]:

# returns the slope of a line
def get_slope(line):
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]
    if (x2-x1)!=0:
        slope = (y2-y1)/(x2-x1)
    else:
        slope=999
    return slope


# In[11]:

# add a point (x_1,y_1) to the intersections list if inside of the rectangle defined
# by the boundaries lower_x,upper_x,lower_y,upper_y
def add_if_within_boundaries(intersections,x_1,y_1,lower_x,upper_x,lower_y,upper_y):
    if ((x_1>=lower_x and x_1<=upper_x) and (y_1>=lower_y and y_1<=upper_y)):
        intersections.append((x_1,y_1))
    return intersections


# In[12]:

# extends a line until it touches the boundaries, keeping the same slope
def extend_line(line, lower_x,upper_x,lower_y,upper_y):
    # todo: verify line is a proper line (e.g. not two equal points)
    intersections = list()
    for x1,y1,x2,y2 in line:
        if (x2-x1)!=0:
            m = (y2-y1)/(x2-x1)
            b = y2-m*x2
            
            if m!=0:
                # here we have to consider 4 cases:
                
                #intersection with lower_y
                x = (lower_y-b)/m
                y = lower_y
                # try to add this one
                intersection = add_if_within_boundaries(intersections,x,y,lower_x,upper_x,lower_y,upper_y)
                
                #intersection with upper_y
                x = (upper_y-b)/m
                y = upper_y
                # try to add this one
                intersection = add_if_within_boundaries(intersections,x,y,lower_x,upper_x,lower_y,upper_y)
                
                #intersection with lower_x
                x = lower_x
                y = m*lower_x+b;
                intersection = add_if_within_boundaries(intersections,x,y,lower_x,upper_x,lower_y,upper_y)
                
                #intersection 2: upper_x
                x = upper_x
                y = m*upper_x+b;
                intersection = add_if_within_boundaries(intersections,x,y,lower_x,upper_x,lower_y,upper_y)
                
            else: # horizontal lines
                x = lower_x
                y = y1# note y1 == y2
                intersection = add_if_within_boundaries(intersections,x,y,lower_x,upper_x,lower_y,upper_y)
                y = y2
                x = upper_x
                intersection = add_if_within_boundaries(intersections,x,y,lower_x,upper_x,lower_y,upper_y)
                
        else: # means vertical lines  OK
            m = 999 #big number
            y = lower_y
            x = x1; # note x1 == x2
            intersection = add_if_within_boundaries(intersections,x,y,lower_x,upper_x,lower_y,upper_y)
            x = x2;
            y = upper_y;
            intersection = add_if_within_boundaries(intersections,x,y,lower_x,upper_x,lower_y,upper_y)

        extended_line = list(set(intersections))
        
        # at this point extended_line should have only 2 points
        slope = get_slope(extended_line);
        
        if extended_line:
            extended_line = [[int(extended_line[0][0]),                              int(extended_line[0][1]),                              int(extended_line[1][0]),                              int(extended_line[1][1])]]
        
    return {'line':extended_line,'slope':slope}


# In[1]:

# warps and image given src and dst points

# enum InterpolationFlags { 
#     INTER_NEAREST = 0, 
#     INTER_LINEAR = 1, 
#     INTER_CUBIC = 2, 
#     INTER_AREA = 3, 
#     INTER_LANCZOS4 = 4, 
#     INTER_MAX = 7, 
#     WARP_FILL_OUTLIERS = 8, 
#     WARP_INVERSE_MAP = 16 
# }
def warp(img, src, dst, back=0):
    img_size = (img.shape[1], img.shape[0])
    
    # compute the perspective transform M
    M = cv2.getPerspectiveTransform(src,dst)
    M_inv = cv2.getPerspectiveTransform(dst,src)
    
    if back:
#         print('warping back')
        #warped = cv2.warpPerspective(img, M_inv, img_size, flags=cv2.INTER_LINEAR) #CV_WARP_FILL_OUTLIERS
        warped = cv2.warpPerspective(img, M_inv, img_size, flags=cv2.INTER_NEAREST)#INTER_MAX INTER_NEAREST #CV_WARP_FILL_OUTLIERS
    else:
        #warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped


# In[14]:

# converts an image to grayscale
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[15]:

# aplly canny filter to image
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


# In[16]:

# apply gaussian blur
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# In[17]:

# crop image to a given region of interes defined by the vertices
def region_of_interest(img, vertices, color_max_value = 255):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (color_max_value,) * channel_count
    else:
        ignore_mask_color = color_max_value
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# In[18]:

# draws a set of lines in the image
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# In[19]:

# computes hough lines from the image
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap);
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8);
    
    lower_x = 0;
    upper_x = img.shape[1];
    lower_y = img.shape[0]*4/7;
    upper_y = img.shape[0];
    # extend lines
    new_lines = list();
    left_lines = list();
    right_lines = list();
    x_left_low = [];
    y_left_low = [];
    x_left_up = [];
    y_left_up = [];
    
    x_right_low = [];
    x_right_up = [];
    y_right_low = [];
    y_right_up = [];
    
    for line in lines:
        result = extend_line(line, lower_x,upper_x,lower_y,upper_y);
        # line now has extra information: the slope (x1, y1, x2, y2, slope)
        new_lines.append(result['line']);
        slope = result['slope'];
        # filter and separate left from right lines
        right_line_min_slope = 0.4;
        right_line_max_slope = 2;
        left_line_min_slope = -2;
        left_line_max_slope = -0.4;
        for x1, y1, x2, y2 in line:
            if (slope >= left_line_min_slope and slope<=left_line_max_slope):
                left_lines.append(line);
                x_left_low.append(x1);
                y_left_low.append(y1);
                x_left_up.append(x2);
                y_left_up.append(y2);
            else: 
                if (slope >= right_line_min_slope and slope<=right_line_max_slope):
                    right_lines.append(line);
                    x_right_low.append(x1);
                    y_right_low.append(y1);
                    x_right_up.append(x2);
                    y_right_up.append(y2);
    
    # average left and right lines
    if (x_left_low and y_left_low and x_left_up and y_left_up):
        
        x_left_low = int(np.mean(x_left_low));
        y_left_low = int(np.mean(y_left_low));
        x_left_up = int(np.mean(x_left_up));
        y_left_up = int(np.mean(y_left_up));

        x_right_low = int(np.mean(x_right_low));
        y_right_low = int(np.mean(y_right_low));
        x_right_up = int(np.mean(x_right_up));
        y_right_up = int(np.mean(y_right_up));
    
        #debug
        #print(x_left_low,',',y_left_low)
        #print(x_left_up,',',y_left_up)
        #print(x_right_low,',',y_right_low)
        #print(x_right_up,',',y_right_up)
        #print ('---------------------')

        left_line = [[x_left_low, y_left_low, x_left_up, y_left_up]];
        right_line = [[x_right_low,y_right_low,x_right_up,y_right_up]];

        # extend lines again
        result = extend_line(left_line, lower_x,upper_x,lower_y,upper_y);
        left_line = result['line'];
        result = extend_line(right_line, lower_x,upper_x,lower_y,upper_y);
        right_line = result['line'];

        left_lines = (left_line,);
        right_lines = (right_line,);

        draw_lines(line_img, left_lines,[0, 255, 0], 10);
        draw_lines(line_img, right_lines,[255, 0, 0], 10);
    
    return line_img


# In[27]:

# returns a weighted image between the two given images 
# note (α + β) <= 1.0, if greater results are incorrect
def weighted_img(img, initial_img, α=0.8, β=0.2, λ=0.):
    """
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# In[28]:

# computes the magnitude of sobel given a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray_im, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_im, cv2.CV_64F, 0, 1,ksize=sobel_kernel)   
    # 3) Calculate the magnitude 
    sobel_mag = np.sqrt((sobelx**2)+(sobely**2))
    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    # 6) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 7) Return this mask as your binary_output image
    return binary_output


# In[29]:

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray_im, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_im, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    gradient_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gradient_dir)
    binary_output[(gradient_dir > thresh[0]) & (gradient_dir < thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


# In[30]:

# computes the abosulte sobel transform
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Grayscale
    #1 convert to gray
    gray_im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply cv2.Sobel()
    if orient=='x':
        sobel = cv2.Sobel(gray_im, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray_im, cv2.CV_64F, 0, 1)   
    # Take the absolute value of the output from cv2.Sobel()
    abs_sobel = np.absolute(sobel)
    # Scale the result to an 8-bit range (0-255)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply lower and upper thresholds
    # Create binary_output
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


# In[1]:

# returns the histogram of a single channel given a threshold
def get_histogram (img_channel,threshold):
    hist = []
    x = []
    height, width = img_channel.shape
    for i in range(width):
        count = 0
        for j in range(height):
            if img_channel[j,i]>threshold:
                count = count + 1
        hist.append(count)
        x.append(i)
    return x,hist


# In[4]:

# get the peak of the histogram for a given image channel and given a threshold
def get_peak(img_channel,threshold=0.5,min_pixels=1):
    # use that region to search for the line in the next section of the image
    x,y = get_histogram(img_channel,threshold)
    # find peaks
    peak_x = y.index(max(y)) #get the x coordinate for the peak on the left
    if max(y) < min_pixels:
        peak_x = int(img_channel.shape[1]/2)
    return peak_x


# In[ ]:



