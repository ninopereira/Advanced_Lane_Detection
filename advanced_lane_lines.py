
# coding: utf-8

# # Advanced Lane Lines Detection
# 
# 
# # Problem
# 
# Given a set of images taken from a camera mounted on top of the car we have to identify the left and right lines of the lane on the road and calculate the radius of curvature. 
# 
# 
# # Proposed Method  
# 
# First we compute the camera calibration matrix and distortion coefficients given a set of chessboard images (in the camera_cal folder in the repository).
# 
# Next, for a series of test images (in the test_images folder in the repository):
# 
# 1. Apply the distortion correction to the raw image.
# 2. Use color transforms, gradients, etc., to create a thresholded binary image.
# 3. Apply a perspective transform to rectify binary image ("birds-eye view").
# 4. Detect lane pixels and fit to find lane boundary.
# 5. Determine curvature of the lane and vehicle position with respect to center.
# 6. Warp the detected lane boundaries back onto the original image.
# 7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 8. Once you have implemented a successful pipeline on the test images, you will run your algorithm on a video. In the case of the video, you must search for the lane lines in the first few frames, and, once you have a high-confidence detection, use that information to track the position and curvature of the lines from frame to frame.
# 9. Check out the project rubric before you submit to make sure your project is complete!.

# In[1]:

# import libs
get_ipython().magic('matplotlib inline')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import copy

# import functions from other files
from camera_calibration import calibrate_camera
from image_transformation import warp
from image_transformation import plotImageSet
from image_transformation import region_of_interest
from image_transformation import get_peak
from image_transformation import weighted_img


# # Helper functions

# In[2]:

# warp a lane shape from the perspective to the orthogonal view
# in fact we want to use it to do the inverse transformation
def warp_lane(img,direction=0):
    if len(img.shape)>2:
        height, width, channels = img.shape
    else:
        height, width = img.shape
        
    # compute the perspective transform M
    src_point_1 = (round(0.22*width),round(0.95*height))
    src_point_2 = (round(width*0.45),round(0.65*height))
    src_point_3 = (round(width*0.58),round(0.65*height))
    src_point_4 = (round(width*0.87), round(0.95*height))

    dst_point_1 = (round(0.2*width),height)
    dst_point_2 = (round(0.2*width),round(0.2*height))
    dst_point_3 = (round(0.85*width),round(0.2*height))
    dst_point_4 = (round(0.85*width),height)

    #four source and destination coordinates
    src = np.float32([src_point_1,src_point_2,src_point_3,src_point_4])
    dst = np.float32([dst_point_1,dst_point_2,dst_point_3,dst_point_4])

    warped_img = warp(img,src,dst,direction)
    return warped_img


# In[3]:

# check and draw warping area of the image
def draw_wrap_area(img, src_points):
    height, width, channels = img.shape
    src_point_1 = (round(0.22*width),round(0.95*height))
    src_point_2 = (round(width*0.45),round(0.65*height))
    src_point_3 = (round(width*0.58),round(0.65*height))
    src_point_4 = (round(width*0.87), round(0.95*height))
    img_wrap_area = copy.copy(img)
    cv2.line(img_bound, src_points[1], src_points[2], (100,0,0), thickness=2)
    cv2.line(img_bound, src_points[2], src_points[3], (100,0,0), thickness=2)
    cv2.line(img_bound, src_points[3], src_points[4], (100,0,0), thickness=2)
    cv2.line(img_bound, src_points[4], src_points[1], (100,0,0), thickness=2)
    return img_wrap_area


# In[4]:

# get list of pixels (x,y coord) for each line:
def get_pixel_lists(im, left_color, right_color):
    left_line_x = []
    left_line_y = []

    right_line_x = []
    right_line_y = []

    print(im.shape)
    height,width,channels = im.shape
    for i in range(height):
         for j in range(width):   
            red_val = im[i,j,0] 
            green_val = im[i,j,1]
            blue_val = im[i,j,2]
            if (red_val == left_color[0] and green_val==left_color[1] and blue_val==left_color[2]): #(255,50,50)
                left_line_y.append(i)
                left_line_x.append(j)
            if (red_val == right_color[0] and green_val==right_color[1] and blue_val==right_color[2]): #(50,150,255)
                right_line_y.append(i)
                right_line_x.append(j)

    left_line_y = np.array(left_line_y)
    left_line_x = np.array(left_line_x)
    right_line_y = np.array(right_line_y)
    right_line_x = np.array(right_line_x)
    return left_line_x,left_line_y,right_line_x,right_line_y


# In[5]:

# fit a 2nd order polynomial to the x and y values
def fit_polynom(x_values, y_values):
    pol_fit = np.polyfit(y_values, x_values, 2)
    pol_fitx = pol_fit[0]*y_values**2 + pol_fit[1]*y_values + pol_fit[2]
    return pol_fitx,pol_fit


# In[6]:

# retrieves the curvature radius (in meters) from the set of points
def get_curvature_radius(x_values,y_values):
    y_eval = np.max(y_values)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    fit_cr  = np.polyfit(y_values*ym_per_pix, x_values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5)                              /np.absolute(2*fit_cr[0])
    return curverad


# In[7]:

# 6. Warp the detected lane boundaries back onto the original image.
# computes the output value given the input value and the 2nd order function (coeficients)
def get_out_value(input, function):
    output = function[0]*input**2 + function[1]*input + function[2]
    return output


# In[8]:

# draw the lane given the edge points of the function
def draw_lane(img,left_line_y,left_fit,right_line_y,right_fit):

    left_y_min = np.min(left_line_y)
    left_y_max = np.max(left_line_y)
    right_y_min = np.min(right_line_y)
    right_y_max = np.max(right_line_y)

    point_1 = round(get_out_value(left_y_min,left_fit)),left_y_min
    point_2 = round(get_out_value(right_y_min,right_fit)),right_y_min
    point_3 = round(get_out_value(right_y_max,right_fit)),right_y_max
    point_4 = round(get_out_value(left_y_max,left_fit)),left_y_max

    polygon = np.array([[[round(get_out_value(left_y_min,left_fit)),left_y_min],
                        [round(get_out_value(right_y_min,right_fit)),right_y_min],
                        [round(get_out_value(right_y_max,right_fit)),right_y_max],
                        [round(get_out_value(left_y_max,left_fit)),left_y_max]]], np.int32)

    image_poly = np.zeros_like(img)
    cv2.fillPoly(image_poly, polygon, (0,255,0))
    inverse_warp = 1
    warped_poly = warp_lane(image_poly,direction=1)
    return warped_poly


# In[16]:

# 4. Detect lane pixels in birds-eye image
# note img_channel should have values ranging from 0 to 255
def identify_lines(img_channel, left_color, right_color):
    thres = 0.5 # threshold to validate line pixels
    original_height, original_width = img_channel.shape
    
    step_height = round(original_height/5)

    left_search_point = round(original_width/4)
    right_search_point = round(3*original_width/4)

    left_amplitude = round(original_width/4) 
    right_amplitude = round(original_width/4)

    line_img = copy.copy(img_channel)
    left_mask = np.zeros_like(img_channel)
    right_mask = np.zeros_like(img_channel)

    for i in range(5):

        img_fraction = img_channel[(original_height-step_height*(i+1)):(original_height-step_height*(i)),
                                   0:original_width]
        height, width = img_fraction.shape

        x1=(left_search_point-left_amplitude)
        x2=(left_search_point+left_amplitude)
        y1=(original_height-step_height*(i+1))
        y2=(original_height-step_height*(i))    

        # validate x1, x2
        x1 = max(x1,0)
        x1 = min(x1,original_width)
        x2 = max(x2,0)
        x2 = min(x2,original_width)
        
        region_left  = img_fraction[0:height,x1:x2]
        
        #(img, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(line_img, (x1, y1), (x2, y2), (255,255,0), 2)
        left_peak_x = x1 + get_peak(region_left,thres)

        x1=(right_search_point-right_amplitude)
        x2=(right_search_point+right_amplitude)
        y1=(original_height-step_height*(i+1))
        y2=(original_height-step_height*(i))

        # validate x1, x2
        x1 = max(x1,0)
        x1 = min(x1,original_width)
        x2 = max(x2,0)
        x2 = min(x2,original_width)
        
        region_right = img_fraction[0:height,x1:x2]
        
        
        cv2.rectangle(line_img, (x1, y1), (x2, y2), (255,255,0), 2)
        right_peak_x = x1 + get_peak(region_right,thres)

        left_search_point = left_peak_x
        right_search_point = right_peak_x
        left_amplitude = 100
        right_amplitude = 100

        #color pixels on the left and on the right

        left_mask[(original_height-step_height*(i+1)):(original_height-step_height*(i)),
                 (left_search_point-left_amplitude):(left_search_point+left_amplitude)] = 1
        right_mask[(original_height-step_height*(i+1)):(original_height-step_height*(i)),
                 (right_search_point-right_amplitude):(right_search_point+right_amplitude)] = 1
        

    r_channel = copy.copy(img_channel)
    g_channel = copy.copy(img_channel)
    b_channel = copy.copy(img_channel)
    im = np.dstack((r_channel, g_channel, b_channel))

    #im = weighted_img(line_img,warped_img) 
    im[(img_channel >= 0.5) & (left_mask >= 0.5)] = left_color
    im[(img_channel >= 0.5) & (right_mask >= 0.5)] = right_color
    
    #plotImageSet([img_channel,line_img,im,])
    return im


# In[10]:

def plot_fit_lines(left_line_x, left_line_y, right_line_x, right_line_y, left_fitx, right_fitx):
    plt.plot(left_line_x, left_line_y, 'o', color='red')
    plt.plot(right_line_x, right_line_y, 'o', color='blue')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, left_line_y, color='green', linewidth=3)
    plt.plot(right_fitx, right_line_y, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images


# # Camera Calibration

# In[11]:

# 0. Calibrate the camera using the calibration images
nx = 9 # number of inside corners in x
ny = 6 # number of inside corners in y
mtx,dist = calibrate_camera('camera_cal/calibration*.jpg',nx, ny)


# # 1 .Apply distortion correction to the raw image.
# 
# # test camera calibration
# fname = 'camera_cal/calibration2.jpg'
# img = mpimg.imread(fname)
# undistorted_im = cv2.undistort(img, mtx, dist, None, mtx)
# plotImageSet([img,undistorted_im]);

# In[12]:

# 2. Use color transforms, gradients, etc., to create a thresholded binary image.


# #0 change color space to HLS
# image_hls = cv2.cvtColor(undistorted_im,cv2.COLOR_RGB2HLS)
# sxbinary = abs_sobel_thresh(undistorted_im,'x',3,(60,155))
# plotImageSet([undistorted_im,sxbinary])

# mag_binary = mag_thresh(undistorted_im, 3,(60,150))
# plotImageSet([undistorted_im,mag_binary])

# dir_binary = dir_threshold(undistorted_im, sobel_kernel=15, thresh=(0.93, 1.04))
# plotImageSet([undistorted_im,dir_binary])

# In[17]:


# Pipeline.
# def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100), r_thresh=(20, 100), h_thresh=(20, 100)):
def pipeline(img, h_thresh=(20,22), s_thresh=(200, 254), sx_thresh=(20, 100),r_thresh=(220, 255)):
    
    img = np.copy(img)
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel hue
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    # Threshold color channel saturation
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    #Threshold color channel r
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
    
    filtered_channel = np.zeros_like(r_channel)
    filtered_channel[(sxbinary >= 0.5) | (s_binary >= 0.5) | (h_binary>= 0.5) | (r_binary>= 0.5)] = 255
    # Stack each channel
    
#     color_binary = np.dstack(( h_binary, sxbinary, s_binary))
    
    # debug
    # h_binary captures yellow lines very well
    # sxbinary captures white lines
    # s_binary captures white and yellow lines
#     plotImageSet([img,h_binary,sxbinary,s_binary,r_binary,filtered_channel])
#     plotImageSet([img,img_lines])
    
    #return color_binary
    return filtered_channel


# In[19]:

# get images list
images = os.listdir("test_images/")

# images = ["solidYellowLeft.jpg",]


# In[20]:

# process all images in the list
for afile in images:
    # get one image from the list
    filename = 'test_images/'+afile
    img = mpimg.imread(filename)
    
    # undistort the image
    undistorted_im = cv2.undistort(img, mtx, dist, None, mtx)
    
    #filter the image
    filtered_channel = pipeline(undistorted_im)
#     plotImageSet([undistorted_im,filtered_channel])
    
    # aplly mask to the region of interest
    # apply 1st mask to select region of interest
    height, width, channels = img.shape
    vertices = np.array([[(0,height),(width*0.45,0.55*height),(width*0.55, 0.55*height), (width, height)]], dtype=np.int32);
    masked_channel = region_of_interest(filtered_channel, vertices);
    # plotImageSet([undistorted_im,masked_channel])

    # warp the image to 'orthogonal bird-eyes view'
    warped_channel = warp_lane(masked_channel)
    # wrap_area = draw_wrap_area(undistorted_im) # just for debug
    # plotImageSet([undistorted_im,wrap_area])
    
#     plotImageSet([warped_channel,])
    # identify lines in image
    left_color = (255,50,50)
    right_color = (50,150,255)
    lines_img = identify_lines(warped_channel, left_color, right_color)
    
    # get pixel list for each line (left and righ)
    left_line_x,left_line_y,right_line_x,right_line_y = get_pixel_lists(lines_img, left_color, right_color)   
    
    # Fit a second order polynomial to each lane line
    left_fitx,left_fit = fit_polynom(left_line_x, left_line_y)
    right_fitx,right_fit = fit_polynom(right_line_x, right_line_y)
    #plot_fit_lines(left_line_x, left_line_y, right_line_x, right_line_y, left_fitx, right_fitx)
    
    # compute and display curvature radius in the image
    left_curverad = get_curvature_radius(left_line_x,left_line_y)
    right_curverad = get_curvature_radius(right_line_x,right_line_y)
    print('left_curverad = ',left_curverad, 'm,  right_curverad =', right_curverad, 'm')

    # 5. Determine the vehicle position with respect to center.
    #ToDo
    
    # draw left and right lines 
    #ToDo
    
    # draw lane
    warped_poly = draw_lane(lines_img,left_line_y,left_fit,right_line_y,right_fit)
    #plotImageSet([img,warped_poly])

    # 7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    processed_img = weighted_img(img,warped_poly);
    
    # show the original and final result
    plotImageSet([img,masked_channel,warped_channel,processed_img])
    
    # save the processed image to the output folder
    filename = 'processed_images/'+afile
    mpimg.imsave(filename,processed_img)


# In[ ]:



