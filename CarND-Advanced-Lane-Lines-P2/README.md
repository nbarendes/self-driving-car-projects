
# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Compute the camera matrix and distortion coefficients.
  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![image](https://user-images.githubusercontent.com/34095574/77430371-5df7ae80-6ddb-11ea-88a3-017cd4a28cf5.png)



### Pipeline 

#### 1. Example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![image](https://user-images.githubusercontent.com/34095574/77431001-5b498900-6ddc-11ea-8814-6378a1b608b3.png)

The pipeline takes two arguments for camera distortion correction. These two arguments are called:
*	mtx: The camera matrix(as returned, i.e. from cv2.calibrateCamera)
*	dist: The camera distortion coefficient(as returned, i.e. from cv2.calibrateCamera)
These two arguments are fed into the cv2.undistort function.


#### 2. Methods to create a thresholded binary image.  

I used a combination of color and gradient thresholds to generate a binary image.  

Step 1: Apply Sobel on x-axis (calculate directional gradient and apply gradient threshold)

Step 2: compute the magnitude of the gradient and apply a threshold

Step 3: computes the direction of the gradient and apply a threshold

Step 4: Apply combination of LUV + LAB + HLS Color threshold ( L channel from LAB color space, B channel from LAB color channel and S           channel from HLS color space)

Step 5: Combine color and gradient thresholds 

```python
def color_gradient_threshold(img):
    # Choose a Sobel kernel size
    ksize = 5 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 200))
    
    #binary_hls = hls_thresh(img, thresh= (170,255))
    binary_luv_lab_hls = LUV_LAB_HLS(img, thresh=(150, 255))
    
    # Combine the two binary thresholds
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) | (binary_luv_lab_hls == 1))] = 1
    
    return combined
```

Results:

![image](https://user-images.githubusercontent.com/34095574/77441983-62779380-6dea-11ea-92d8-32552899b852.png)


#### 3. Perspective transform.

The code for my perspective transform includes a function called `warp()`, which appears in the cell #22 of the IPython notebook.  The `warp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
height,width = img.shape[:2]
width_Offset = 240
height_Offset = 0
img_size = (img.shape[1], img.shape[0])

src = np.float32([[(592, 450), (180, height), (1130, height), (687, 450)]])
dst = np.float32([[(width_Offset, height_Offset), (width_Offset, height), (1040, height), (1040, height_Offset)]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 592, 450      | 240, 0        | 
| 180, 720      | 240, 720      |
| 1130, 720     | 1040, 720      |
| 687, 450      | 1040, 0        |

Draw the src and dst points onto a test image and its warped counterpart to verify that perspective transform works as expected. Lines appear parallel in the warped image:


![image](https://user-images.githubusercontent.com/34095574/77444541-4248d400-6dec-11ea-8e69-760810f8b056.png)

Example with warped and masked image:

![image](https://user-images.githubusercontent.com/34095574/77444729-7f14cb00-6dec-11ea-90cf-8a09c445d6e8.png)



#### 4. Identify lane-line pixels and fit their positions with a polynomial.


***Step 1***: Find the starting point for the left and right lines (take a histogram of the bottom half of the masked image)

```python
def hist(top_view):
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = top_view[top_view.shape[0]//2:,:]
    # The highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram
```

![image](https://user-images.githubusercontent.com/34095574/77446517-cdc36480-6dee-11ea-9fcf-761267b235de.png)

The next steps is to initiate a Sliding Window Search in the left and right parts which we got from the histogram.

The sliding window is applied in following steps:


***Step 2***: We then calculate the position of all non zero x and non zero y pixels.

***Step 3***: Start iterating over the windows where we start from points calculate in point 1.

***Step 4***: identify the non zero pixels in the window we just defined.

***Step 5***: collect all the indices in the list and decide the center of next window using these points.

***Step 6***: seperate the points to left and right positions.

***Step 7***: fit a second degree polynomial using np.polyfit and point calculate in step 6.

```python
def fit_polynomial(binary_warped, show = False):
    # Find our lane pixels first
        # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        if show == True:
        # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
        else:
            pass
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    if show == True:
        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
    else:
        pass

    return out_img,left_fitx, right_fitx, leftx, lefty, rightx, righty, ploty,left_fit, right_fit
  
 ```

![image](https://user-images.githubusercontent.com/34095574/77467906-168a1600-6e0d-11ea-926d-778249b56268.png)



#### 5. Calculated the radius of curvature of the lane , the position of the vehicle with respect to center and conversion from pixel to real world.

I did this in cell #34 in the fuction `measure_curvature_pixels()`

Following this conversion, we can now compute the radius of curvature at any point x on the lane line represented by the function x = f(y) as follows:

![image](https://user-images.githubusercontent.com/34095574/77525914-0cf2c380-6e8a-11ea-91fe-51c6068fa8e0.png)


In the case of the second order polynomial above, the first and second derivatives are:

![image](https://user-images.githubusercontent.com/34095574/77526164-707cf100-6e8a-11ea-8e17-388a1512d7f8.png)

So, our equation for radius of curvature becomes:

![image](https://user-images.githubusercontent.com/34095574/77529467-f5b6d480-6e8f-11ea-9fc3-72d9cdc5a5a0.png)


Note: since the y-values for an image increases from top to bottom, we compute the lane line curvature at y = img.shape[0], which is the point closest to the vehicle.

To report the lane line curvature in metres we first need to convert from pixel space to real world space. For this, we measure the width of a section of lane that we're projecting in our warped image and the length of a dashed line. Once measured, we compare our results with the U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines length of 3.048 meters.




#### 6. Result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell #40  in the function `process()`. 

Here is an example of my result on a test image:

![image](https://user-images.githubusercontent.com/34095574/77470104-70400f80-6e10-11ea-9c38-bb305f72bc2c.png)


---

### Pipeline (video)

#### 1. Link to the final video output.


Here's a [link to my video result](https://youtu.be/OFwiNzpElJk)

---

### Discussion



The first problem is to find the correct source and destination points. It is a hit and trial approach . The second problem is when I was trying to use various combinations of color channels the final combination did not work in almost all conditions. It was again by hit and trial I figured out bad frames and checked my pipleline and made changes to and/or operators and thresholds. The next challenge and the biggest problem is to stop flickering of lane lines on concrete surface or when the car comes out from the shadow.

I tried my pipeline on the challenge video and I noticed it failed. So I will be experimenting with the challenge video for sure. It is quite possible that left lane line to center is of different color and from center to right lane is of different color as in the challenge video and it is likely to fail there. Also in case of a mountain terrain, it is quite likely to fail.

To make it more robust and stop the flickering of lane lines, we can average out the points from the previous frames to have a smooth transition per frame.
