
# coding: utf-8

# This notebook contains imported code from project 1.
# 
# Here we have a collection of usefull functions to do image transformation

# In[10]:

get_ipython().magic('matplotlib inline')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


# In[11]:

#display images side by side
def plotImageSet(image_list,color='gnuplot2'):
    fig = plt.figure()
    count = 1
    for image in image_list:
        ax = fig.add_subplot(1,len(image_list),count)
        imgplot = plt.imshow(image,cmap=color)
        ax.axis('off')
        count = count +1
    plt.show()


# In[12]:

def warp(img, src, dst, back=0):
    img_size = (img.shape[1], img.shape[0])
    
    # compute the perspective transform M
    M = cv2.getPerspectiveTransform(src,dst)
    M_inv = cv2.getPerspectiveTransform(dst,src)
    
    if back:
        print('warping back')
        warped = cv2.warpPerspective(img, M_inv, img_size, flags=cv2.INTER_LINEAR)
    else:
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


# In[13]:

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

def add_if_within_boundaries(intersections,x_1,y_1,lower_x,upper_x,lower_y,upper_y):
    if ((x_1>=lower_x and x_1<=upper_x) and (y_1>=lower_y and y_1<=upper_y)):
        intersections.append((x_1,y_1))
    return intersections
    
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


# In[14]:

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
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
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


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

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

