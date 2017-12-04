# **Project 1: Finding Lane Lines on the Road** 


---

**Target of this project**
---
![alt text][image0]

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

This report is composed with the following parts:
* Describing my pipeline and techniques.
* Display my results, including pictures and videos.
* Identify potential shortcomings with my pipeline.
* Possible improvements that can be made in the future.

[//]: # (Image References)

[image0]: ./test_images_output/solidWhiteCurve_after.jpg "solidWhiteCurve_after"
[image0_1]: ./test_images_output_NoAverageExtrap/solidWhiteCurve_after.jpg "solidWhiteCurve_after_noExtrap"
[image1]: ./pipeline_images/1_displayed_image.png "1_displayed_image"
[image2]: ./pipeline_images/2_grayscale_image.png "2_grayscale_image"
[image3]: ./pipeline_images/3_gauss_smoth.png "3_gauss_smoth"
[image4]: ./pipeline_images/4_canny_edge.png "4_canny_edge"
[image5]: ./pipeline_images/5_interested_region.png "5_interested_region"
[image6]: ./pipeline_images/6_draw_houghline.png "6_draw_houghline"
[image7]: ./pipeline_images/7_line_on_original_image.png "7_line_on_original_image"
[image11a]: ./test_images/solidWhiteCurve.jpg
[image11b]: ./test_images_output/solidWhiteCurve_after.jpg
[image12a]: ./test_images/solidWhiteRight.jpg
[image12b]: ./test_images_output/solidWhiteRight_after.jpg
[image13a]: ./test_images/solidYellowCurve.jpg
[image13b]: ./test_images_output/solidYellowCurve_after.jpg
[image14a]: ./test_images/solidYellowCurve2.jpg
[image14b]: ./test_images_output/solidYellowCurve2_after.jpg
[image15a]: ./test_images/solidYellowLeft.jpg
[image15b]: ./test_images_output/solidYellowLeft_after.jpg
[image16a]: ./test_images/whiteCarLaneSwitch.jpg
[image16b]: ./test_images_output/whiteCarLaneSwitch_after.jpg

---


### 1. The Pipeline to find lane markers.

My pipeline consisted of 5 steps:

#### 1. Convert the images to grayscale.
First of all, we need to load and take a look at the test images before doing anything.
I used the following code to see what test images I have:
```python
import os
os.listdir("test_images/")
```
The output is:
```python
['solidWhiteCurve.jpg',
 'solidWhiteRight.jpg',
 'solidYellowCurve.jpg',
 'solidYellowCurve2.jpg',
 'solidYellowLeft.jpg',
 'whiteCarLaneSwitch.jpg']
 ```
 So I select one of them to display using the following code:
 ```python
# Read in image and show
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
 ```
Then we can see the image is:

 ![alt text][image1]
 
Apply Greyscale function on it:
```python
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
 ![alt text][image2]

#### 2. Apply Gaussian smoothing on the images.
Blur the grayscale image:
```python
blur_gray = gaussian_blur(gray,5)
plt.imshow(blur_gray)
```
where gaussian_blur function is define as:
```python
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```
get the output as:

 ![alt text][image3]

#### 3. Apply Canny edge detection to detect the edge points in the images, and use a set of points to define a polygon area as region of interest.
Canny edge detection function is given as:
```python
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
```
Use it to detect the edges in the blurred grayscale image:
```python
edges = canny(blur_gray, 50, 150)
plt.imshow(edges)
```
Here, the parameter **low_threshold** and **high_threshold** are tuned to get the best result.
Output is here:

 ![alt text][image4]

In order to focus on the lane markers, we need to isolate the region where the lane markers are in. Using **region_of_interest** function to to this:
```python
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
```
The key variable **vertices** is tuned according to the shape of the image:
```python
imshape = image.shape
print(imshape)
vertices = np.array([[(100,imshape[0]),(465, 320), (510, 320), (imshape[1]-40,imshape[0])]], dtype=np.int32)
masked_edges = region_of_interest(edges, vertices)
plt.imshow(masked_edges)
```
Then the edge points representing lane markers are selected:

 ![alt text][image5]


#### 4. Perform Hough Transform on the selected edges to find lines; average and extrapolate the lines and plot them on the original images.

Hough Transform is used to find the points that form a line in the image, this part of pipeline is shown in the cell below. Furthermore, in the **hough_lines** function, we are calling three important functions to **average**, **extrapolate** and **draw** the lines. These 3 functions are explained after this cell.
```python
# Perform Hough Transform
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 100 #minimum number of pixels making up a line
max_line_gap = 200    # maximum gap in pixels between connectable line segments
line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
plt.imshow(line_image)
```
* The main function used here, **hough_lines function**, is defined as:
```python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    left_mark_para,right_mark_para = average_lines(lines) # to get the averaged slope and intercept of lines
    
    y_bottom = img.shape[0] # y value of bottom of image, i.e. bottom of the lane marker
    y_top = int(0.6*img.shape[0])
    
    left_lanemark = extrap_func(y_bottom,y_top,left_mark_para) # bottom and top end points of left lane marker, after extrapolation and averaging
    right_lanemark = extrap_func(y_bottom,y_top,right_mark_para)
    ## left_lanemark = tuple(map(tuple, left_lanemark))
    ## right_lanemark = tuple(map(tuple, right_lanemark))
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    draw_lines(line_img, (left_lanemark,right_lanemark))
    return line_img
```
* First of all, **cv2.HoughLinesP** is used to find all the lines that satisfy the conditions defined by minLineLength and maxLineGap. This function returns the variable **lines**.
* Then, the function **average_lines** is utilized to get the averaged slope and intercept of lines:
```python
def average_lines(lines):
    """
    Note: this function is to average the lines 
    """
    left_lines = []
    left_length = []
    right_lines = []
    right_length = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1==x2:
                continue
            if y1==y2:
                continue
            
            slope = (y2-y1)/(x2-x1)
            intercept = y1-slope*x1
            length = np.sqrt((y1-y2)**2+(x1-x2)**2)
            if slope>0: # right side
                right_lines.append((slope, intercept))
                right_length.append((length))
            else:
                left_lines.append((slope, intercept))
                left_length.append((length))
                
    # to average lines, add more weight on longer lines to take weighted average
    left_linesAve = np.dot(left_length, left_lines)/np.sum(left_length) if len(left_length)>0 else None
    right_linesAve = np.dot(right_length, right_lines)/np.sum(right_length) if len(right_length)>0 else None
    
    return left_linesAve, right_linesAve # each one contains a pair of averaged slope and intercept
```
* Then, we need to extrapolate the averaged lines, otherwise, the lines will look like:

 ![alt text][image0_1]
 
 * The extrapolation function is defined as follows:
 ```python
 def extrap_func(y1,y2,line_para):
    """
    this function is to approach the end points (x1,y1) and (x2,y2) of a line using given slope and intercept
    the input line_para should be left_linesAve and right_linesAve from average_lines function
    """
    if line_para is None:
        return None
    
    slope, intercept = line_para
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1,y1),(x2,y2))
 ```
 * Finally, the **draw_lines** function is called to draw the lines on the original image:
 ```python
 def draw_lines(img, lines, color=[255, 0, 0], thickness=15):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    for line in lines:
        if line is not None:
            cv2.line(img, *line, color, thickness)
        # disabled the following code to draw extrapolated averaged lines only
        # for x1,y1,x2,y2 in line:
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
 ```
Eventually, we get the extrapolated and averaged lines on a blank image:

 ![alt text][image6]
 
 After this, we need to take one additional step to overlap the lines with the original image using the *weight_image* function:
```python
original_pic_with_lines = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
plt.imshow(original_pic_with_lines)
```
This time we get the final output:

 ![alt text][image7]

#### 5. Apply steps 1 to 5 on all six test images to make sure it works for different cases, then apply this pipeline on the videos.

First of all, we need to make sure this pipeline works well on videos which are nothing but a series of images. Therefore, we need to double check the output of the pipeline on images with different characteristics. Six different images have been provided as test images. I processed them with my pipeline, and got the following results:

Original Images        |    Images with detected lines
:---------------------:|:-----------------------------:
 ![alt text][image11a] |  ![alt text][image11b]
 ![alt text][image12a] |  ![alt text][image12b]
 ![alt text][image13a] |  ![alt text][image13b]
  ![alt text][image14a] |  ![alt text][image14b]
  ![alt text][image15a] |  ![alt text][image15b]
  ![alt text][image16a] |  ![alt text][image16b]

Everything seems good! Now we can apply the pipeline to the videos.
Before we do that, we need to stack the entire pipeline in one function for processing video:
```python
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    
    # TODO: put your pipeline here,
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray,5)
    edges = canny(blur_gray, 50, 150)
    imshape = image.shape
    vertices = np.array([[(100,imshape[0]),(465, 320), (510, 320), (imshape[1]-40,imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    # Perform Hough Transform
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 #minimum number of pixels making up a line
    max_line_gap = 200    # maximum gap in pixels between connectable line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    # you should return the final output (image where lines are drawn on lanes)
    original_pic_with_lines = weighted_img(line_image, image, α=0.8, β=1., λ=0.)

    return original_pic_with_lines
```
Now we can try the videos!
First we try the solidYellowLeft.mp4...
```python
white_output = 'test_videos_output/solidYellowLeft.mp4'

## Use the following line for a shorter subclip, use next line for entire video
## clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,4)
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```
Run this cell, we get a processed video. Then we can play it in the Jupyter Notebook with this cell:
```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```
I've uploaded the videos to YouTube. The links for the processed videos are:

[1. Processed solidYellowLeft](https://www.youtube.com/watch?v=QQP3TZniM74)

[2. Processed solidWhiteRight](https://www.youtube.com/watch?v=gxK5LMZL3VM)

### 2. Potential shortcomings with the current pipeline

One potential shortcoming would be what would happen when the road lanes are curved. Since the current pipeline is based on the Hough Transform to detect lines, it will encounter problems when lane markers are not straight.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use more advanced techniques to detect edge points that form a curve and decribe them with polynomial parameters.

