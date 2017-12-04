# **Project 1: Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**
---
![alt text][image1]

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

This report is composed with the following parts:
* Describing my pipeline and techniques.
* Display my results, including pictures and videos.
* Identify potential shortcomings with my pipeline.
* Possible improvements that can be made in the future.

[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve_after.jpg "solidWhiteCurve_after"

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
 

#### 2. Apply Gaussian smoothing on the images.

#### 3. Apply Canny edge detection to detect the edge points in the images, and use a set of points to define a polygon area as region of interest.

#### 4. Perform Hough Transform on the selected edges to find lines.

#### 5. Average and extropolate the lines and  plot them on the original images. Then apply this pipeline on the videos.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by make the following changes:

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=15):
    """    
    this function draws lines
    """
    
    for line in lines:
        if line is not None:
            cv2.line(img, *line, color, thickness)
        # disabled the following code to draw extrapolated averaged line
        # for x1,y1,x2,y2 in line:
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
```


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
