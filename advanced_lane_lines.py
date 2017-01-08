
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

# In[106]:

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

# In[107]:

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


# In[108]:

# check and draw warping area of the image
def draw_wrap_area(img):
    height, width, channels = img.shape
    src_point_1 = (round(0.22*width),round(0.95*height))
    src_point_2 = (round(width*0.45),round(0.65*height))
    src_point_3 = (round(width*0.58),round(0.65*height))
    src_point_4 = (round(width*0.87), round(0.95*height))
    img_wrap_area = copy.copy(img)
    cv2.line(img_wrap_area, src_point_1, src_point_2, (100,0,0), thickness=2)
    cv2.line(img_wrap_area, src_point_2, src_point_3, (100,0,0), thickness=2)
    cv2.line(img_wrap_area, src_point_3, src_point_4, (100,0,0), thickness=2)
    cv2.line(img_wrap_area, src_point_4, src_point_1, (100,0,0), thickness=2)
    return img_wrap_area


# In[109]:

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


# In[110]:

# fit a 2nd order polynomial to the x and y values
def fit_polynom(x_values, y_values):
    pol_fit = np.polyfit(y_values, x_values, 2)
    pol_fitx = pol_fit[0]*y_values**2 + pol_fit[1]*y_values + pol_fit[2]
    return pol_fitx,pol_fit


# In[111]:

# retrieves the curvature radius (in meters) from the set of points
def get_curvature_radius(x_values,y_values):
    y_eval = np.max(y_values)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    fit_cr  = np.polyfit(y_values*ym_per_pix, x_values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5)                              /np.absolute(2*fit_cr[0])
    return curverad


# In[112]:

# 6. Warp the detected lane boundaries back onto the original image.
# computes the output value given the input value and the 2nd order function (coeficients)
def get_out_value(input, function):
    output = function[0]*input**2 + function[1]*input + function[2]
    return output


# # draw the lane given the edge points of the function
# def draw_lane(img,left_line_y,left_fit,right_line_y,right_fit):
# 
#     left_y_min = np.min(left_line_y)
#     left_y_max = np.max(left_line_y)
#     right_y_min = np.min(right_line_y)
#     right_y_max = np.max(right_line_y)
# 
#     point_1 = round(get_out_value(left_y_min,left_fit)),left_y_min
#     point_2 = round(get_out_value(right_y_min,right_fit)),right_y_min
#     point_3 = round(get_out_value(right_y_max,right_fit)),right_y_max
#     point_4 = round(get_out_value(left_y_max,left_fit)),left_y_max
# 
#     polygon = np.array([[[round(get_out_value(left_y_min,left_fit)),left_y_min],
#                         [round(get_out_value(right_y_min,right_fit)),right_y_min],
#                         [round(get_out_value(right_y_max,right_fit)),right_y_max],
#                         [round(get_out_value(left_y_max,left_fit)),left_y_max]]], np.int32)
# 
#     image_poly = np.zeros_like(img)
#     cv2.fillPoly(image_poly, polygon, (0,255,0))
#     inverse_warp = 1
#     warped_poly = warp_lane(image_poly,direction=1)
#     return warped_poly

# In[113]:

# draw the lane given the edge points of the function
def draw_lane(img,left_fit,right_fit,start_y=0,end_y=0):
    if end_y==0:
        end_y = img.shape[2]
        
#     image_poly = np.zeros_like(img)
#     image_poly = copy.copy(img)
    color = (0,255,0)
    #draw line from left to right
    for y_val in range(start_y,end_y-1):
        pt1 = (int(get_out_value(y_val, left_fit)),y_val)
        pt2 = (int(get_out_value(y_val, right_fit)),y_val)
        cv2.line(img,pt1,pt2,color)
    
    
    #cv2.fillPoly(image_poly, polygon, (0,255,0))
#     inverse_warp = 1
#     warped_poly = warp_lane(img,direction=1)
#     return warped_poly
    return img


# In[114]:

# 4. Detect lane pixels in birds-eye image
# note img_channel should have values ranging from 0 to 255
def identify_lines(img_channel, left_color, right_color, debug=0):
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

        lx1=(left_search_point-left_amplitude)
        lx2=(left_search_point+left_amplitude)
        ly1=(original_height-step_height*(i+1))
        ly2=(original_height-step_height*(i))    

        # validate x1, x2
        lx1 = max(lx1,0)
        lx1 = min(lx1,original_width)
        lx2 = max(lx2,0)
        lx2 = min(lx2,original_width)
        
        region_left  = img_fraction[0:height,lx1:lx2]
        
        #(img, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(line_img, (lx1, ly1), (lx2, ly2), (255,255,0), 2)
        left_peak_x = lx1 + get_peak(region_left,thres)

        rx1=(right_search_point-right_amplitude)
        rx2=(right_search_point+right_amplitude)
        ry1=(original_height-step_height*(i+1))
        ry2=(original_height-step_height*(i))

        # validate x1, x2
        rx1 = max(rx1,0)
        rx1 = min(rx1,original_width)
        rx2 = max(rx2,0)
        rx2 = min(rx2,original_width)
        
        region_right = img_fraction[0:height,rx1:rx2]
        
        
        cv2.rectangle(line_img, (rx1, ry1), (rx2, ry2), (255,255,0), 2)
        right_peak_x = rx1 + get_peak(region_right,thres)

        left_search_point = left_peak_x
        right_search_point = right_peak_x
        left_amplitude = 100
        right_amplitude = 100

        #color pixels on the left and on the right
        left_mask[(original_height-step_height*(i+1)):(original_height-step_height*(i)),lx1:lx2] = 1
        right_mask[(original_height-step_height*(i+1)):(original_height-step_height*(i)),rx1:rx2] = 1
        
        if debug:
            plt.imshow(line_img,'gray')
            plt.show()

    r_channel = copy.copy(img_channel)
    g_channel = copy.copy(img_channel)
    b_channel = copy.copy(img_channel)
    im = np.dstack((r_channel, g_channel, b_channel))
    
    im[(img_channel >= 0.5) & (left_mask >= 0.5)] = left_color
    im[(img_channel >= 0.5) & (right_mask >= 0.5)] = right_color
    return im


# In[115]:

def draw_line(img,fit_func,color=(255,255,255), thickness=24):
    height, width, channels = img.shape
    pixels = []
    # get the pixels to color
    for y_val in range(height):
        x_val = get_out_value(y_val,fit_func)
        pixels.append([y_val,x_val])
    # color the pixels
    for pixel in pixels:
        for i in range(thickness):
            img[pixel[0]][int(pixel[1]-thickness/2+i)]=color
        
    return img


# In[116]:

def plot_fit_lines(left_line_x, left_line_y, right_line_x, right_line_y, left_fitx, right_fitx):
    plt.plot(left_line_x, left_line_y, 'o', color='red')
    plt.plot(right_line_x, right_line_y, 'o', color='blue')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, left_line_y, color='green', linewidth=3)
    plt.plot(right_fitx, right_line_y, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images


# # Camera Calibration

# In[117]:

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

# In[118]:

# 2. Use color transforms, gradients, etc., to create a thresholded binary image.


# #0 change color space to HLS
# image_hls = cv2.cvtColor(undistorted_im,cv2.COLOR_RGB2HLS)
# sxbinary = abs_sobel_thresh(undistorted_im,'x',3,(60,155))
# plotImageSet([undistorted_im,sxbinary])

# mag_binary = mag_thresh(undistorted_im, 3,(60,150))
# plotImageSet([undistorted_im,mag_binary])

# dir_binary = dir_threshold(undistorted_im, sobel_kernel=15, thresh=(0.93, 1.04))
# plotImageSet([undistorted_im,dir_binary])

# In[128]:


# Pipeline.
# def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100), r_thresh=(20, 100), h_thresh=(20, 100)):
def pipeline(img, h_thresh=(20,22), s_thresh=(200, 254), sx_thresh=(40, 100),r_thresh=(220, 255)):
    
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
    plotImageSet([img,h_binary,sxbinary,s_binary,r_binary,filtered_channel])
#     plotImageSet([img,img_lines])
    
    #return color_binary
    return filtered_channel


# In[142]:

# get images list
images = os.listdir("test_images/")

# images = ["solidYellowLeft.jpg",]
# images = ["test1.jpg"]


# In[144]:

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
    wrap_area = draw_wrap_area(undistorted_im) # just for debug
    
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
    
    fit_lines_img = np.zeros_like(lines_img)
    fit_lines_img = draw_lane(fit_lines_img,left_fit,right_fit,start_y=0,end_y=height)#round(height/5)
    fit_lines_img = draw_line(fit_lines_img,left_fit,left_color)
    fit_lines_img = draw_line(fit_lines_img,right_fit,right_color)
    
    # warp image back
    warped_poly = warp_lane(fit_lines_img,direction=1)

    # 5. Determine the vehicle position with respect to center.
    #ToDo
    
    # 7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    processed_img = weighted_img(warped_poly,img,0.5);
    
    # compute and display curvature radius in the image
    left_curverad = round(get_curvature_radius(left_line_x,left_line_y))
    right_curverad = round(get_curvature_radius(right_line_x,right_line_y))
    print('left_curverad = ',left_curverad, 'm,  right_curverad =', right_curverad, 'm')
    lcurverad_txt = 'left curvature = '+str(round(left_curverad))+' m'
    rcurverad_txt = 'right curvature = '+str(round(right_curverad))+' m'
#     curverad_txt = 'radius of curvature = '+str(round((left_curverad+right_curverad)/2))+' m'
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(processed_img,curverad_txt,(10,500), font, 1,(255,255,255),2)
    cv2.putText(processed_img,lcurverad_txt, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(processed_img,rcurverad_txt, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    # show the original and final result
    plotImageSet([img,masked_channel,lines_img,processed_img])
    
    # save the processed image to the output folder
    filename = 'processed_images/'+afile
    mpimg.imsave(filename,processed_img)


# In[ ]:



