# advanced_lane_lines

## Problem

The goal is to write a software pipeline to identify the lane boundaries in video from a front-facing camera on a car. The camera calibration images, test road images, and videos were given a priori.

## Method
To complete the project the following steps were taken:

1) compute the camera calibration matrix and distortion coefficients given a set of chessboard images;

2) for a series of test images:
  Apply the distortion correction to the raw image.
  Use color transforms, gradients, etc., to create a thresholded binary image.
  Apply a perspective transform to rectify binary image ("birds-eye view").
  Detect lane pixels and fit to find lane boundary.
  Determine curvature of the lane and vehicle position with respect to center.
  Warp the detected lane boundaries back onto the original image.
  Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

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
If another picture following the previous one is to be searched for lines, we can use the knowledge to narrow down the search into a smaller region (**Posterior lower searches**).

| Initial lower search | Posterior lower searches |
| --- | --- |
| <img src="report_img/id_lines_00.png" width="200"/> | <img src="report_img/id_lines_0.png" width="200"/> |

In one case or the other the subsequent steps are the same: we search for the next segment of the line starting from the 

| id_lines_0 | id_lines_1 | id_lines_2 | id_lines_3 | id_lines_4 |
| --- | --- | --- | --- | --- |
| <img src="report_img/id_lines_0.png" width="100"/> | <img src="report_img/id_lines_1.png" width="100"/> | | <img src="report_img/id_lines_2.png" width="100"/> | <img src="report_img/id_lines_3.png" width="100"/> | | <img src="report_img/id_lines_4.png" width="100"/> | 
| id_lines_5 | id_lines_6 | id_lines_7 | id_lines_8 | id_lines_9 |
 | --- | --- | --- | --- | --- |
| <img src="report_img/id_lines_5.png" width="100"/> | | <img src="report_img/id_lines_6.png" width="100"/> | | <img src="report_img/id_lines_7.png" width="100"/> | | <img src="report_img/id_lines_8.png" width="100"/> | <img src="report_img/id_lines_9.png" width="100"/> | 

Have lane line pixels been identified in the rectified image and fit with a polynomial?

Methods have been used to identify lane line pixels in the rectified binary image. The left and right line have been identified and fit with a curved functional form (e.g., spine or polynomial).

Having identified the lane lines, has the radius of curvature of the road been estimated? And the position of the vehicle with respect to center in the lane?

Here the idea is to take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature may be given in meters assuming the curve of the road follows a circle and the position of the vehicle within the lane may be given as meters off of center.

Has the result from lane line detection been warped back to the original image space and displayed?

The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This should demonstrate that the lane boundaries were correctly identified.

Pipeline (video)

CRITERIA
MEETS SPECIFICATIONS
Does the pipeline established with the test images work to process the video?

The image processing pipeline that was established to find the lane lines in images successfully processes the video. The output here should be a new video where the lanes are identified in every frame, and outputs are generated regarding the radius of curvature of the lane and vehicle position within the lane. The identification and estimation don't need to be perfect, but they should not be wildly off in any case. The pipeline should correctly map out curved lines and not fail when shadows or pavement color changes are present.

Has some kind of search method been implemented to discover the position of the lines in the first images in the video stream?

In the first few frames of video, the algorithm should perform a search without prior assumptions about where the lines are (i.e., no hard coded values to start with). Once a high-confidence detection is achieved, that positional knowledge may be used in future iterations as a starting point to find the lines.

Has some form of tracking of the position of the lane lines been implemented?

As soon as a high confidence detection of the lane lines has been achieved, that information should be propagated to the detection step for the next frame of the video, both as a means of saving time on detection and in order to reject outliers (anomalous detections).

Readme

CRITERIA
MEETS SPECIFICATIONS
Has a Readme file been included that describes in detail the steps taken to construct the pipeline, techniques used, areas where improvements could be made?

The Readme file submitted with this project includes a detailed description of what steps were taken to achieve the result, what techniques were used to arrive at a successful result, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.
