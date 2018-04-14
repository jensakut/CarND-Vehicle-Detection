## Vehicle detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/dataset.png
[image2]: ./output_images/spatial.png
[image3]: ./output_images/HOG.png
[image4]: ./output_images/windows.png
[image5]: ./output_images/pipeline_boxes.jpg
[image6]: ./output_images/pipeline_results_green.jpg
[image7]: ./output_images/pipeline_labels.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook in the function get_hog_features: 

```python
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with tconvert_colorwo outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are examples of both classes:

![alt text][image1]

In order to perform a classification, the picture is transformed into a vector. The vectors for different colorspaces applied to both classes are depicted here: 

![alt text][image2]

Visually, it is difficult to spot any difference. Same goes for the HOG-transformation. The picture shows the saturation-channel of an HSV image, with 9 orientations, 8 pixels per cell and 2 cells per block. 

![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of color-parameters.

The visual exploration of the dataset is interesting, yet the performance of the classifier matters most. Splitting the dataset and looping through the parameter sets is performed in [the vehicle detection parameter exploration](https://github.com/jensakut/CarND-Vehicle-Detection/blob/master/Vehicle_detection_parameter_exploration.ipynb) 

| Color space | Spatial, hist   | Training duration s | Training samples | Prediction time | Accuracy % |
|-------------|-----------------|---------------------|------------------|-----------------|------------|
| RGB | 32 , 32 | 47.94 | 14208 | 0.00451 | 0.9079 |
| RGB | 32 , 32 | 47.06 | 14208 | 0.01354 | 0.9091 |
| HSV | 32 , 32 | 30.11 | 14208 | 0.01351 | 0.9032 |
| HSV | 32 , 32 | 28.31 | 14208 | 0.00501 | 0.8919 |
| LUV | 32 , 32 | 24.55 | 14208 | 0.00853 | 0.9361 |
| LUV | 32 , 32 | 21.68 | 14208 | 0.00852 | 0.9372 |
| HLS | 32 , 32 | 25.71 | 14208 | 0.00501 | 0.9071 |
| HLS | 32 , 32 | 25.32 | 14208 | 0.00802 | 0.904 |
| YUV | 32 , 32 | 33.31 | 14208 | 0.00251 | 0.911 |
| YUV | 32 , 32 | 33.96 | 14208 | 0.01153 | 0.8978 |
| YCrCb | 32 , 32 | 0.14 | 14208 | 0.00904 | 0.4952 |
| YCrCb | 32 , 32 | 0.14 | 14208 | 0.02106 | 0.5011 |
| LAB | 32 , 32 | 0.14 | 14208 | 0.0 | 0.5068 |
| LAB | 32 , 32 | 0.14 | 14208 | 0.00953 | 0.5011 |
| RGB | 16 , 16 | 17.2 | 14208 | 0.00302 | 0.9305 |
| RGB | 16 , 16 | 16.59 | 14208 | 0.002 | 0.9257 |
| HSV | 16 , 16 | 16.28 | 14208 | 0.00251 | 0.9093 |
| HSV | 16 , 16 | 15.68 | 14208 | 0.00251 | 0.911 |
| LUV | 16 , 16 | 10.46 | 14208 | 0.00201 | 0.9485 |
| LUV | 16 , 16 | 10.4 | 14208 | 0.00201 | 0.9409 |
| HLS | 16 , 16 | 16.33 | 14208 | 0.00201 | 0.9133 |
| HLS | 16 , 16 | 15.75 | 14208 | 0.00201 | 0.9136 |
| YUV | 16 , 16 | 15.5 | 14208 | 0.00251 | 0.9251 |
| YUV | 16 , 16 | 15.77 | 14208 | 0.00201 | 0.9206 |
| YCrCb | 16 , 16 | 0.05 | 14208 | 0.0015 | 0.5011 |
| YCrCb | 16 , 16 | 0.05 | 14208 | 0.00201 | 0.5149 |
| LAB | 16 , 16 | 0.05 | 14208 | 0.0015 | 0.5059 |
| LAB | 16 , 16 | 0.05 | 14208 | 0.00351 | 0.5011 |


LUV combines speed and precision in both parameter variations. All tests are performed twice to ensure results are reproducible. In the following, each parameter combination will be performed once. 


| Color space | Spatial, hist   | Training duration s | Training samples | Prediction time | Accuracy % |
|-------------|-----------------|---------------------|------------------|-----------------|------------|
| LUV | 4 , 4 | 3.23 | 14208 | 0.00249 | 0.8466 |
| LUV | 4 , 8 | 2.46 | 14208 | 0.00201 | 0.9068 |
| LUV | 4 , 12 | 2.72 | 14208 | 0.0015 | 0.9248 |
| LUV | 4 , 16 | 2.84 | 14208 | 0.0015 | 0.9206 |
| LUV | 4 , 20 | 2.48 | 14208 | 0.002 | 0.9254 |
| LUV | 4 , 24 | 3.15 | 14208 | 0.00101 | 0.9265 |
| LUV | 4 , 32 | 3.1 | 14208 | 0.001 | 0.926 |
| LUV | 8 , 4 | 5.53 | 14208 | 0.001 | 0.9133 |
| LUV | 8 , 8 | 4.55 | 14208 | 0.00201 | 0.9423 |
| LUV | 8 , 12 | 4.53 | 14208 | 0.002 | 0.9437 |
| LUV | 8 , 16 | 4.91 | 14208 | 0.00201 | 0.9488 |
| LUV | 8 , 20 | 4.62 | 14208 | 0.00201 | 0.9558 |
| LUV | 8 , 24 | 4.31 | 14208 | 0.00201 | 0.9513 |
| LUV | 8 , 32 | 4.38 | 14208 | 0.00201 | 0.9521 |
| LUV | 12 , 4 | 10.11 | 14208 | 0.00149 | 0.9307 |
| LUV | 12 , 8 | 8.0 | 14208 | 0.00152 | 0.949 |
| LUV | 12 , 12 | 7.97 | 14208 | 0.0015 | 0.9524 |
| LUV | 12 , 16 | 7.23 | 14208 | 0.0015 | 0.9516 |
| LUV | 12 , 20 | 7.24 | 14208 | 0.0015 | 0.9569 |
| LUV | 12 , 24 | 7.0 | 14208 | 0.00201 | 0.9589 |
| LUV | 12 , 32 | 6.45 | 14208 | 0.0015 | 0.9533 |
| LUV | 16 , 4 | 13.35 | 14208 | 0.0015 | 0.9316 |
| LUV | 16 , 8 | 10.48 | 14208 | 0.00251 | 0.942 |
| LUV | 16 , 12 | 10.12 | 14208 | 0.0015 | 0.9555 |
| LUV | 16 , 16 | 10.09 | 14208 | 0.00301 | 0.9513 |
| LUV | 16 , 20 | 9.3 | 14208 | 0.00201 | 0.9474 |
| LUV | 16 , 24 | 8.51 | 14208 | 0.00199 | 0.9521 |
| LUV | 16 , 32 | 8.3 | 14208 | 0.00201 | 0.9606 |
| LUV | 20 , 4 | 17.0 | 14208 | 0.002 | 0.9282 |
| LUV | 20 , 8 | 10.9 | 14208 | 0.00251 | 0.9305 |
| LUV | 20 , 12 | 11.08 | 14208 | 0.00451 | 0.9299 |
| LUV | 20 , 16 | 10.1 | 14208 | 0.00201 | 0.9268 |
| LUV | 20 , 20 | 10.51 | 14208 | 0.00501 | 0.9369 |
| LUV | 20 , 24 | 8.99 | 14208 | 0.00251 | 0.9395 |
| LUV | 20 , 32 | 9.14 | 14208 | 0.00251 | 0.9417 |
| LUV | 24 , 4 | 18.65 | 14208 | 0.00251 | 0.9071 |
| LUV | 24 , 8 | 14.97 | 14208 | 0.00351 | 0.9212 |
| LUV | 24 , 12 | 14.74 | 14208 | 0.0015 | 0.9288 |
| LUV | 24 , 16 | 13.3 | 14208 | 0.00201 | 0.924 |
| LUV | 24 , 20 | 14.05 | 14208 | 0.00352 | 0.9268 |
| LUV | 24 , 24 | 13.15 | 14208 | 0.00251 | 0.9316 |
| LUV | 24 , 32 | 11.31 | 14208 | 0.00201 | 0.9333 |
| LUV | 32 , 4 | 30.35 | 14208 | 0.01955 | 0.9068 |
| LUV | 32 , 8 | 29.81 | 14208 | 0.00953 | 0.9271 |
| LUV | 32 , 12 | 26.65 | 14208 | 0.01404 | 0.9251 |
| LUV | 32 , 16 | 26.46 | 14208 | 0.01554 | 0.9271 |
| LUV | 32 , 20 | 26.6 | 14208 | 0.02106 | 0.9322 |
| LUV | 32 , 24 | 24.79 | 14208 | 0.00501 | 0.9299 |
| LUV | 32 , 32 | 24.46 | 14208 | 0.00803 | 0.9392 |

Using spatial size of 16 and 32 histogram bins provides best accuracy, yet 8 / 20 costs way less computational effort. 

### HOG parameter search

First, the combinations of color-spaces are being evaluated. Some colorspaces produced issues getting converted to gradients and were discarded as a result. 

|Colorspace|HOG-Channel|Orientations|Pixel per Cell|Cell per Block|Feature vector length|Extraction Duration|Training duration|Prediction Duration|Accuracy
|----------|------------|--------------|--------------|-----------|---------------------|-------------------|-----------------|---------------|---------|
| RGB | 0 | 9 | 8 | 2 | 1764 | 47.8 | 26.2 | 0.00451 | 0.9037 |
| RGB | 1 | 9 | 8 | 2 | 1764 | 48.35 | 19.06 | 0.00251 | 0.9186 |
| RGB | 2 | 9 | 8 | 2 | 1764 | 49.43 | 18.28 | 0.00247 | 0.9068 |
| RGB | ALL | 9 | 8 | 2 | 5292 | 134.74 | 43.56 | 0.00752 | 0.9237 |
| HSV | 0 | 9 | 8 | 2 | 1764 | 54.51 | 29.03 | 0.00201 | 0.8908 |
| HSV | 1 | 9 | 8 | 2 | 1764 | 44.77 | 25.47 | 0.00201 | 0.913 |
| HSV | 2 | 9 | 8 | 2 | 1764 | 46.28 | 17.91 | 0.00202 | 0.913 |
| HSV | ALL | 9 | 8 | 2 | 5292 | 129.46 | 30.37 | 0.00301 | 0.96 |
| LUV | 0 | 9 | 8 | 2 | 1764 | 54.18 | 17.93 | 0.0015 | 0.9153 |
| HLS | 0 | 9 | 8 | 2 | 1764 | 48.52 | 29.88 | 0.004 | 0.8891 |
| HLS | 1 | 9 | 8 | 2 | 1764 | 48.37 | 20.39 | 0.00301 | 0.9071 |
| HLS | 2 | 9 | 8 | 2 | 1764 | 49.98 | 28.79 | 0.0015 | 0.904 |
| HLS | ALL | 9 | 8 | 2 | 5292 | 137.89 | 33.38 | 0.00401 | 0.9569 |
| YUV | 0 | 9 | 8 | 2 | 1764 | 56.21 | 18.48 | 0.00201 | 0.9181 |
| YUV | 1 | 9 | 8 | 2 | 1764 | 46.14 | 16.71 | 0.00199 | 0.9243 |
| YCrCb | 0 | 9 | 8 | 2 | 1764 | 45.29 | 18.17 | 0.00201 | 0.9257 |
| YCrCb | 1 | 9 | 8 | 2 | 1764 | 48.28 | 19.51 | 0.003 | 0.9237 |
| YCrCb | 2 | 9 | 8 | 2 | 1764 | 47.65 | 27.69 | 0.002 | 0.9003 |
| YCrCb | ALL | 9 | 8 | 2 | 5292 | 130.52 | 24.25 | 0.00451 | 0.9597 |
| LAB | 0 | 9 | 8 | 2 | 1764 | 59.36 | 21.36 | 0.0025 | 0.9229 |

Judging by these results, the following combinations are chosen to be of interest: 
|Colorspace|HOG-Channel|
|----------|------------|
|HSV | ALL|
|HLS | ALL|
|YCrCb | ALL| 
|LAB | 0|
|YCrCB | 0|  
|YCrCB | 1|
|YUV | 1|

A quick evaluation of 500 samples each provided these results: 

|Colorspace|HOG-Channel|Orientations|Pixel per Cell|Cell per Block|Feature vector length|Extraction Duration|Training duration|Prediction Duration|Accuracy
|----------|------------|--------------|--------------|-----------|---------------------|-------------------|-----------------|---------------|---------|
| HSV | ALL | 3 | 4 | 2 | 8100 | 16.85 | 4.6 | 0.00201 | 0.88 |
| HSV | ALL | 3 | 8 | 2 | 1764 | 4.99 | 0.97 | 0.00149 | 0.915 |
| HSV | ALL | 3 | 16 | 2 | 324 | 2.41 | 0.15 | 0.002 | 0.93 |
| HSV | ALL | 9 | 8 | 2 | 5292 | 5.38 | 3.14 | 0.00201 | 0.865 |
| HSV | ALL | 9 | 16 | 2 | 972 | 2.9 | 0.46 | 0.001 | 0.925 |
| HSV | ALL | 12 | 4 | 2 | 32400 | 17.02 | 12.02 | 0.00197 | 0.855 |
| HSV | ALL | 12 | 8 | 2 | 7056 | 5.81 | 2.63 | 0.00201 | 0.885 |
| HSV | ALL | 12 | 16 | 2 | 1296 | 3.08 | 0.63 | 0.00151 | 0.865 |
| HLS | ALL | 3 | 4 | 2 | 8100 | 16.47 | 4.51 | 0.00251 | 0.9 |
| HLS | ALL | 3 | 8 | 2 | 1764 | 4.97 | 1.06 | 0.00251 | 0.905 |
| HLS | ALL | 3 | 16 | 2 | 324 | 2.43 | 0.17 | 0.0015 | 0.945 |
| HLS | ALL | 9 | 4 | 2 | 24300 | 16.42 | 7.33 | 0.00201 | 0.87 |
| HLS | ALL | 9 | 8 | 2 | 5292 | 5.57 | 3.01 | 0.00251 | 0.865 |
| HLS | ALL | 9 | 16 | 2 | 972 | 2.78 | 0.44 | 0.00101 | 0.935 |
| HLS | ALL | 12 | 4 | 2 | 32400 | 16.73 | 8.93 | 0.00251 | 0.905 |
| HLS | ALL | 12 | 8 | 2 | 7056 | 5.9 | 4.91 | 0.0015 | 0.875 |
| HLS | ALL | 12 | 16 | 2 | 1296 | 3.22 | 0.66 | 0.00149 | 0.905 |
| YCrCb | ALL | 3 | 4 | 2 | 8100 | 16.84 | 1.19 | 0.00201 | 0.94 |
| YCrCb | ALL | 3 | 8 | 2 | 1764 | 4.81 | 0.78 | 0.002 | 0.925 |
| YCrCb | ALL | 3 | 16 | 2 | 324 | 2.3 | 0.1 | 0.002 | 0.945 |
| YCrCb | ALL | 9 | 4 | 2 | 24300 | 16.91 | 1.12 | 0.0015 | 0.975 |
| YCrCb | ALL | 9 | 8 | 2 | 5292 | 5.48 | 0.65 | 0.00201 | 0.935 |
| YCrCb | ALL | 9 | 16 | 2 | 972 | 2.75 | 0.37 | 0.0015 | 0.93 |
| YCrCb | ALL | 12 | 4 | 2 | 32400 | 17.27 | 1.24 | 0.00201 | 0.965 |
| YCrCb | ALL | 12 | 8 | 2 | 7056 | 5.74 | 0.64 | 0.00199 | 0.95 |
| YCrCb | ALL | 12 | 16 | 2 | 1296 | 3.04 | 0.54 | 0.002 | 0.91 |
| LAB | 0 | 3 | 4 | 2 | 2700 | 5.76 | 2.39 | 0.0015 | 0.82 |
| LAB | 0 | 3 | 8 | 2 | 588 | 2.15 | 0.4 | 0.00199 | 0.885 |
| LAB | 0 | 3 | 16 | 2 | 108 | 1.27 | 0.11 | 0.00251 | 0.895 |
| LAB | 0 | 9 | 8 | 2 | 1764 | 2.33 | 1.26 | 0.00201 | 0.865 |
| LAB | 0 | 9 | 16 | 2 | 324 | 1.43 | 0.19 | 0.00151 | 0.89 |
| LAB | 0 | 12 | 4 | 2 | 10800 | 6.6 | 5.52 | 0.0015 | 0.86 |
| LAB | 0 | 12 | 8 | 2 | 2352 | 2.62 | 1.96 | 0.002 | 0.85 |
| LAB | 0 | 12 | 16 | 2 | 432 | 1.63 | 0.23 | 0.0015 | 0.9 |
| YCrCb | 0 | 3 | 4 | 2 | 2700 | 6.1 | 2.47 | 0.00151 | 0.815 |
| YCrCb | 0 | 3 | 8 | 2 | 588 | 2.02 | 0.39 | 0.00152 | 0.82 |
| YCrCb | 0 | 3 | 16 | 2 | 108 | 1.11 | 0.12 | 0.0015 | 0.89 |
| YCrCb | 0 | 9 | 4 | 2 | 8100 | 6.5 | 6.08 | 0.0015 | 0.79 |
| YCrCb | 0 | 9 | 8 | 2 | 1764 | 2.23 | 1.43 | 0.00149 | 0.83 |
| YCrCb | 0 | 9 | 16 | 2 | 324 | 1.28 | 0.19 | 0.001 | 0.92 |
| YCrCb | 0 | 12 | 4 | 2 | 10800 | 6.41 | 6.02 | 0.00201 | 0.8 |
| YCrCb | 0 | 12 | 8 | 2 | 2352 | 2.32 | 2.11 | 0.00201 | 0.83 |
| YCrCb | 0 | 12 | 16 | 2 | 432 | 1.36 | 0.23 | 0.001 | 0.93 |
| YUV | 1 | 3 | 4 | 2 | 2700 | 6.18 | 0.51 | 0.00151 | 0.855 |
| YUV | 1 | 3 | 8 | 2 | 588 | 2.1 | 0.28 | 0.0015 | 0.885 |
| YUV | 1 | 3 | 16 | 2 | 108 | 1.1 | 0.09 | 0.0015 | 0.95 |
| YUV | 1 | 9 | 4 | 2 | 8100 | 6.32 | 0.36 | 0.002 | 0.855 |
| YUV | 1 | 9 | 8 | 2 | 1764 | 2.19 | 0.3 | 0.0015 | 0.905 |
| YUV | 1 | 9 | 16 | 2 | 324 | 1.27 | 0.14 | 0.001 | 0.905 |
| YUV | 1 | 12 | 4 | 2 | 10800 | 6.46 | 0.43 | 0.0015 | 0.915 |
| YUV | 1 | 12 | 8 | 2 | 2352 | 2.31 | 0.38 | 0.00351 | 0.905 |
| YUV | 1 | 12 | 16 | 2 | 432 | 1.35 | 0.17 | 0.001 | 0.89 |


|Colorspace|HOG-Channel|Orientations|Pixel per Cell|Cell per Block|Feature vector length|Extraction Duration|Training duration|Prediction Duration|Accuracy
|----------|------------|--------------|--------------|-----------|---------------------|-------------------|-----------------|---------------|---------|
| HSV | ALL | 3 | 16 | 2 | 324 | 2.41 | 0.15 | 0.002 | 0.93 |
| HSV | ALL | 9 | 16 | 2 | 972 | 2.9 | 0.46 | 0.001 | 0.925 |
| HLS | ALL | 3 | 16 | 2 | 324 | 2.43 | 0.17 | 0.0015 | 0.945 |
| HLS | ALL | 9 | 16 | 2 | 972 | 2.78 | 0.44 | 0.00101 | 0.935 |
| YCrCb | ALL | 3 | 4 | 2 | 8100 | 16.84 | 1.19 | 0.00201 | 0.94 |
| YCrCb | ALL | 3 | 8 | 2 | 1764 | 4.81 | 0.78 | 0.002 | 0.925 |
| YCrCb | ALL | 3 | 16 | 2 | 324 | 2.3 | 0.1 | 0.002 | 0.945 |
| YCrCb | ALL | 9 | 4 | 2 | 24300 | 16.91 | 1.12 | 0.0015 | 0.975 |
| YCrCb | ALL | 9 | 8 | 2 | 5292 | 5.48 | 0.65 | 0.00201 | 0.935 |
| YCrCb | ALL | 9 | 16 | 2 | 972 | 2.75 | 0.37 | 0.0015 | 0.93 |
| YCrCb | ALL | 12 | 4 | 2 | 32400 | 17.27 | 1.24 | 0.00201 | 0.965 |
| YCrCb | ALL | 12 | 8 | 2 | 7056 | 5.74 | 0.64 | 0.00199 | 0.95 |
| YCrCb | ALL | 12 | 16 | 2 | 1296 | 3.04 | 0.54 | 0.002 | 0.91 |
| YCrCb | 0 | 9 | 16 | 2 | 324 | 1.28 | 0.19 | 0.001 | 0.92 |
| YCrCb | 0 | 9 | 16 | 2 | 324 | 1.28 | 0.19 | 0.001 | 0.92 |
| YCrCb | 0 | 12 | 16 | 2 | 432 | 1.36 | 0.23 | 0.001 | 0.93 |
| YUV | 1 | 3 | 16 | 2 | 108 | 1.1 | 0.09 | 0.0015 | 0.95 |
| YUV | 1 | 12 | 4 | 2 | 10800 | 6.46 | 0.43 | 0.0015 | 0.915 |


|Colorspace|HOG-Channel|Orientations|Pixel per Cell|Cell per Block|Feature vector length|Extraction Duration|Training duration|Prediction Duration|Accuracy
|----------|------------|--------------|--------------|-----------|---------------------|-------------------|-----------------|---------------|---------|
| HSV | ALL | 3 | 16 | 2 | 324 | 53.83 | 5.24 | 0.00201 | 0.9572 |
| HSV | ALL | 9 | 16 | 2 | 972 | 51.38 | 4.67 | 0.0015 | 0.9724 |
| HLS | ALL | 3 | 16 | 2 | 324 | 44.27 | 4.82 | 0.0015 | 0.9468 |
| HLS | ALL | 9 | 16 | 2 | 972 | 50.6 | 4.8 | 0.002 | 0.9671 |
| YCrCb | ALL | 3 | 4 | 2 | 8100 | 293.59 | 41.87 | 0.00451 | 0.9505 |
| YCrCb | ALL | 3 | 8 | 2 | 1764 | 113.01 | 7.84 | 0.00351 | 0.9679 |
| YCrCb | ALL | 3 | 16 | 2 | 324 | 62.34 | 3.08 | 0.00199 | 0.978 |
| YCrCb | ALL | 9 | 4 | 2 | 24300 | 319.12 | 209.69 | 0.01164 | 0.9519 |
| YCrCb | ALL | 9 | 8 | 2 | 5292 | 170.86 | 24.58 | 0.00401 | 0.9642 |
| YCrCb | ALL | 9 | 16 | 2 | 972 | 87.12 | 3.6 | 0.0015 | 0.9792 |
| YCrCb | ALL | 12 | 4 | 2 | 32400 | 341.52 | 196.52 | 0.01504 | 0.964 |
| YCrCb | ALL | 12 | 8 | 2 | 7056 | 115.06 | 36.26 | 0.00551 | 0.9651 |
| YCrCb | ALL | 12 | 16 | 2 | 1296 | 60.63 | 4.3 | 0.00251 | 0.9792 |
| YCrCb | 0 | 9 | 16 | 2 | 324 | 26.58 | 5.91 | 0.00351 | 0.9426 |
| YCrCb | 0 | 9 | 16 | 2 | 324 | 26.03 | 6.15 | 0.00201 | 0.9409 |
| YUV | 1 | 3 | 16 | 2 | 108 | 22.84 | 3.99 | 0.0015 | 0.9448 |
| YUV | 1 | 3 | 16 | 2 | 108 | 22.84 | 3.99 | 0.0015 | 0.9448 |
| YUV | 1 | 12 | 4 | 2 | 10800 | 119.91 | 85.1 | 0.00602 | 0.9051 |

Finally, the YCrCb 0-channel was chosen because it combined speed with reasonable single-channel performance. Benchmarking the highest precision YCrCb - all in combination with LUV 16 - 16 enabled an accuracy of 99.6 %, yet the fastest combination still reaches 99.4 % at half the computational effort. 
Thus, YCrCb 0-channel with 9 orientations and 16 pixels per cell where combined with LUV-Channel and 8 spatial size with 20 histogram bins. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The parameters of the feature extraction are as follows: 

```python
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (8, 8) # Spatial binning dimensions
hist_bins = 20    # Number of histogram binscolor_space_HOG = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

color_space_HOG='YCrCb'
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
orient = 9  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block

spatial_feat = True # Spatial features on or off
hist_feat = True#Histogram features on or off
hog_feat = True # HOG features on or off
```

I trained a linear SVM using the parameter-search function GridSearchCV. Thus, I changed parameters to kernel = rbf and C = 5. 


```python
    print('Start searching2')

    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 20]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)

    print("The best parameters are: {}".format(clf.best_params_))
```



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding windows were implemented in this code. The smallest windows are supposed to scan the smaller 64x64 pixel images search for distant cars. To save some computational effort, the x-range is limited. The medium-sized pixels search a wider area, and the biggest 128x128 windows cover the lower half of the screen up to the car's hood. 

```python
    def slide_windows(self, shape):
        self.slide_window(shape, x_start_stop=[250,1030], y_start_stop=[400,500], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))  
        self.slide_window(shape, x_start_stop=[None,None], y_start_stop=[400,500], 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        self.slide_window(shape, x_start_stop=[None, None], y_start_stop=[450,600], 
                     xy_window=(128, 128), xy_overlap=(0.6, 0.6))
        print('search_windows: ',len(self.window_list))
```

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The output of the pipeline is shown in the following visualization: 

![alt text][image5]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_videos/project_video.mp4)

It may be found on Youtube: 
<a href="http://www.youtube.com/watch?feature=player_embedded&v=ID8P9ivLqn4
" target="_blank"><img src="http://img.youtube.com/vi/ID8P9ivLqn4/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

```python
        # create empty single channel to draw heat on
        heat= np.zeros_like(frame[:,:,0]).astype(np.float)
        # every box adds +1 heat to corresponding pixels
        heat = self.add_heat(heat, hot_windows)
        # remove everything below or at threshold and add it to the list
        self.heat.append(self.apply_threshold(heat,1))
        if len(self.heat)>20:
            self.heat.pop(0)
        heat=np.array(np.sum(self.heat, axis=0))
        # points need to be in at least three consecutive frames
        heat=self.apply_threshold(heat,len(self.heat)/2)
        # use label to find common heatspots
        labels = label(heat)
```



### Here are six frames in grayscale (green) and their corresponding heatmaps (red):

![alt text][image6]


### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
It is possible to improve the classifier by additional data augmentation, hard negative mining, classifier parameters tuning etc.
I guess the pipeline is relatively unstable against weather, reflection, shadows, or tight roads. 
The part of the picture which need to be searched is relatively small. Searching the image where cars appear and then searching around their last known position might improve the algorithm a lot. A kalman-filter may be introduced to track vehicles even if cars are overlapping each other. 
The saved data points may be invested into a higher framerate. 
 It will be interesting to use the pipeline on a homemade video. 
I guess a neural net may perform better at extracting the features. Yet, comparing the colorspaces and the parameters was quite interesting. It may be interesting whether this parametersearch is relevant for a neural net, because the computational effort of training a svm is considerably less. 

