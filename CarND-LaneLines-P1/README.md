# **Finding Lane Lines on the Road**



---

### **1. Pipeline**

The goal of this project is to identify lane lines on the road from a dashcam video.

The steps of my pipeline of this project are the following:

1. I converted RGB to HLS colour space.The lines are more recognizable in HLS color space.

2. I converted the HLS color space image to gray scale (single channel) for edge detection.
  
3. I used bilateral_filter to smoothen the image.

4. I used Canny Edge Detector to find out the edges of the lanes.

5. I selected the region of interest with a polygon.

6. I used Hough Line Transform to find out the lines in ROI.

7. I extrapolate the lines as a single lane line. I used the weighted average method.Firstly i separated the right and left lane by        using slope(the left lane should have a positive slope, and the right lane should have a negative slope.) then weighted average the      slope and intercept of left and right lane then i extrapolate the lines.

8. I draw the lanes on the image and used the weighted image of line and the original image. 






---




### 2.Potential shortcomings 

1. The ROI needs to be dynamic, as there will be cases when the lanes will not be perfectly fit in the fixed ROI.

2. Hough Lines based on straight lines do not work good for curved road/lane.

3. Region of Interest assumes that camera stays at same location and lanes are flat. 

4. There are many trial-and-error to get hyper parameters correct. 

5. If one or more lines are missing the pipeline don't work properly.



### 3. Possible improvements

1. Instead of line , it would be beneficial to use higher degree curve that will be useful on curved road.

2. If a line is not detected, we could estimate the current slope using the previous estimations and/or the other line detection

3. Update the ROI mask dynamically
