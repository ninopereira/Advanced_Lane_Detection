# Advanced Lane Detection

![Sample](report_img/curvature_lane.png)


## Problem

The goal is to write a software pipeline to identify the lane boundaries in video from a front-facing camera on a car. The camera calibration images, test road images, and videos were given a priori.

## Method
To complete the project the following steps were taken:

1) compute the camera calibration matrix and distortion coefficients given a set of chessboard images;

2) for a series of test images:
  - Apply the distortion correction to the raw image.
  - Use color transforms, gradients, etc., to create a thresholded binary image.
  - Apply a perspective transform to rectify binary image ("birds-eye view").
  - Detect lane pixels and fit to find lane boundary.
  - Determine curvature of the lane and vehicle position with respect to center.
  - Warp the detected lane boundaries back onto the original image.
  - Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Implementation details

### Camera Calibration
The camera matrix and distortion coefficients were computed and checked on the calibration test image:
The process makes use of opencv library, namely the following functions:
 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 - cv2.findChessboardCorners
 - cv2.drawChessboardCorners
 - cv2.calibrateCamera
 - cv2.undistort
Here's an example of a distorted image then undistorted by applying the undistort method:  

| Distorted Image | Undistorted Image |  
| --- | --- |  
| <img src="report_img/distorted.png" width="300"/>  | <img src="report_img/undistorted.png" width="300"/> |  


### Pipeline (single images)

#### Step 1: Undistort the image
The distortion correction previously calculated is applied to each image.  

<img src="report_img/original_img.png" width="300"/> 

#### Step 2: Filter the image
A binary image has been created using color transforms and gradients?
First the original image is split into separate rgb channels and also hls channels
The Sobel gradient is calculated and only the x component is used as it captures better the vertical lines on the image.
Also r_channel was used as it captures yellow lines and also white lines very efficiently.
Other attempts were tested but for some or other reasons they would inflict noisy data in the process. In general here are the main findings:
- h_binary (hue channel) captures yellow lines very well but introduces noise
- sxbinary (sobel x gradient) captures vertical white lines
- s_binary (saturation channel) captures white and yellow lines but introduces noise when there are shadows
- r_binary (red channel) captures yellow lines very well  

In the end only the **sxbinary** and **r_binary** were combined to produce a filtered binary image containing likely lane pixels.
 
| h_binary | sxbinary |  s_binary | r_binary | filtered |
| --- | --- | --- | --- | --- |
| <img src="report_img/h_binary.png" width="150"/> | <img src="report_img/sxbinary.png" width="150"/> | <img src="report_img/s_binary.png" width="150"/> | <img src="report_img/r_binary.png" width="150"/> | <img src="report_img/filtered_channel.png" width="150"/> |

#### Step 3: Aplly mask to the region of interest
By analysing the image when a long straight road is ahead we derived the region of interest and masked the image so that no external objects, like sky and trees interfere with line detection.

| filtered image |  masked filtered image |
| --- | --- |
| <img src="report_img/filtered_channel.png" width="300"/> | <img src="report_img/masked_channel.png" width="300"/> |

#### Step 4: Warp the image to 'orthogonal bird-eyes view'

An orthogonal 'birds eye view' of the road enables us to better distinguish the lines in the image, since they will appear straight almost vertical and both lines would show up parallel.

| warped image |
| --- |
| <img src="report_img/warped_channel.png" width="300"/> |

#### Step 5: Identify lines in image

When searching for lines in a filtered image we first search the first lower half of the image by splitting it further into left and right. The first detection (**Initial lower search**) searches for left and right line independently, expecting to find them respectively in the left and right parts of the image.

| Initial lower search | 
| --- | 
| <img src="report_img/id_lines_00.png" width="400"/> |
| <img src="report_img/left_hist.png" width="200"/> <img src="report_img/right_hist.png" width="200"/> |

In one case or the other the subsequent 10 steps are the same: we search for the next segment of the line starting from the x location in the image where the previous histogram peak was found and move upwards one step at a time, marking all pixels found inside the search area (marked as rectangles in the pictures below). 


| 1 | 2 | 3 | 4 | 5 |  
| --- | --- | --- | --- | --- |  
| <img src="report_img/id_lines_0.png" width="150"/> | <img src="report_img/id_lines_1.png" width="150"/> | <img src="report_img/id_lines_2.png" width="150"/> | <img src="report_img/id_lines_3.png" width="150"/> | <img src="report_img/id_lines_4.png" width="150"/> |  
| 6 | 7 | 8 | 9 | 10 |  
| --- | --- | --- | --- | --- |  
| <img src="report_img/id_lines_5.png" width="150"/> | <img src="report_img/id_lines_6.png" width="150"/> | <img src="report_img/id_lines_7.png" width="150"/> | <img src="report_img/id_lines_8.png" width="150"/> | <img src="report_img/id_lines_9.png" width="150"/> | 

If another image in the video sequence following the previous one is to be searched for lines, we can use the previous knowledge about the location of the initial peaks to narrow down the search into a smaller region (**Posterior lower searches**) starting already at that location and with a narrower amplitude.

| Posterior lower searches |
| --- |
| <img src="report_img/id_lines_0.png" width="300"/> |

From here the process goes on as before.

#### Step 6: Get the pixels lists and fit polynomials to left and right lines

After detecting and marking all the pixels for the left and right lines, we fit a polynomial to each of them and then use the polynomials to draw the lines and color in green the area in between. After that this image is warped back using the inverse transformation used before when converting to the birds eye view, this time, the function outputs a warped image in perpective view.

| left and right pixels marked | polynomial fitted lines |  warped back img |
| --- | --- | --- |
| <img src="report_img/lines_img.png" width="200"/> | <img src="report_img/fit_lines_img.png" width="200"/> |  <img src="report_img/warped_poly.png" width="200"/> | 
 
#### Step 7: Superimpose the detected lines in the original undistorted image

This step is just the superimposition of the detected lines and lane with the original undistorted image.

| undistorted img | processed image |
| --- | --- | 
| <img src="report_img/original_img.png" width="300"/> | <img src="report_img/processed_img.png" width="300"/> |  

#### Step 8: Calculate the radius of curvature and position of the vehicle

Using the polinomials for each line we calculated the radius of curvature for each line independently. If the difference between both radiuses is less than 30% (empirical value), then we average them and output as lane curvature. If they disagree in more than 30%, then we just print both in the output panel.
The curvatures are displayed in meters.

The position of the car with respect to the center of the lane is calculated and displayed in the final processed and anotated image (mid-lane distance). The position is displayed in meters off the center.

| curvature of the lane | curvature of each line |
| --- | --- | 
| <img src="report_img/curvature_lane.png" width="400"/> | <img src="report_img/curvature_lines.png" width="400"/> |  

By visual inspection we can see that the lane was correctly identified in this image using the proposed pipeline.  


### Pipeline (video)  

The pipeline described above was used to process a sequence of images from a video file.
The video for the project is properly annotated. The output video correctly identifies and displays the lanes correctly across all the images in the video.
The pipeline correctly maps out curved lines and doesn't fail when shadows or pavement color changes are present.

Result
[Video output](https://youtu.be/_5ln3s9YjC8)

#### Discover the position of the lines in the first images in the video stream  

To take advantage of the continued streaming of images from the camera, we should take into consideration the location of detected lines in the previous image and use that information to narrow down the search for lines in the next image of the sequence (see 'Posterior lower searches' in step 5 of the pipeline). This way, we not only speed up the process of detection but we also avoid erroneous detection of unwanted elements that might be in other locations of the image.

In order to do that properly it was implemented a mechanism to allow for a given number of lines to be detected properly and only when the detection is stable, the process of narrowing down the search begins.
The implementation is basically just a filter using a weighted average of the previous detections with the current one.
You can find the related variables in the code:  

```
process_image.left_sp_avg
process_image.right_sp_avg
process_image.count
```
#### Binary image enhancement from consecutive frames  

Moving one step furthe in taking advantage of the continued streaming of images, two consecutive filtered images were merged in order to produce a more robust detection of the lines. As it can be seen below the previous filtered image is merged with the current one to give a slightly better binary image for processing. This way it smoothes out some filtering imperfections and also diminishes the probality of detecting other strange objects in the image as lines.

| previous filtered image | current filtered image | merged image |
| --- | --- | --- |
| <img src="report_img/previous.png" width="300"/> | <img src="report_img/next.png" width="300"/> | <img src="report_img/merged.png" width="300"/> |  

As it can be seen, the image on the right (merged image) has longer lines, hence it is less prone to failures.


## Further Work

A successful implementation of computer vision methods allows for a smooth and consistent lane detection on the project video file. The pipeline has some robustness and was designed to minimise the chances of misdetection in the particular test images and project video files provided.
As such, it is foreseable that using this pipeline without further refinement would not suit all other input images we can provide it (e.g. the challenge video files). Other methods would have to be added in order to improve robustness and broader scope to the current implementation. 

### Challenging scenarios  

Some challenging scenarios would be for example:
- changing lighting conditions
- sudden appearance of other objects (e.g. cars) in the image
- absence of lines in one or the other side
- car changing lanes

### Suggestions for improvements  

One could mitigate some of the problems by further improving the pipeline:
- Create a robust filter which takes into account the past images of the video file
- Detect other objects in the image that might be obstructing the lines.
- Bound the ratio of curvature to admissible values and use that to detect a failure case
- Use the speed and direction of the vehicle in conjunction with the sequence of images to predict the location of line pixels the next frame taking into account the current and previous locations.
