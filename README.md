# project1

### Links
[Color Seg tutorial doc](https://umd.zoom.us/j/95317373459?pwd=Ynp0aStLaURGNjBobGd5SVZwQTlsdz09)

[Color Seg tutorial vod](https://www.youtube.com/watch?v=D5AcaFMY_BI&feature=youtu.be&t=5)

[Project Description](https://cmsc426.github.io/2020/proj/p1/)

[Train_Images](https://drive.google.com/file/d/17XiM86JqHqko4JC00-E4w4sPKnzh2iMz/view?usp=sharing)

###Deadline
11:59:59 PM, September 22, 2022 

We will release the test dataset 48 hours before the deadline i.e. 11:59:59PM, Sunday, September 20.

###Problem Statement
- Write MATLAB code to cluster the orange ball using Single Gaussian [30 points]
- Write MATLAB code to cluster the orange ball using Gaussian Mixture Model [40 points] and 
- Estimate the distance to the ball [20 points]. 
- Also, plot all the GMM ellipsoids [10 points].

### File tree and naming
Your submission on Canvas must be a zip file, following the naming convention YourDirectoryID_proj1.zip. For example, xyz123_proj1.zip. The file must have the following directory structure.

YourDirectoryID_proj1.zip.

- train_images/.
- test_images/.
- results/.
- gaussian.m (For Single Gaussian)
- GMM.m
- trainGMM.m
- testGMM.m
- measureDepth.m
- plotGMM.m
- report.pdf

### starter code
![GMM](https://cmsc426.github.io/assets/proj1/proj1_image.PNG)

### report
For each section of the project, explain briefly what you did, and describe any interesting problems you encountered and/or solutions you implemented. You must include the following details in your writeup:

Your choice of color space, initialization method and number of gaussians in the GMM
- Explain why GMM is better than single gaussian
- Present your distance estimate and cluster segmentation results for each test image
- Explain strengths and limitations of your algorithm. Also, explain why the algorithm failed on some test images
As usual, your report must be full English sentences, not commented code. There is a word limit of 1500 words and no minimum length requirement