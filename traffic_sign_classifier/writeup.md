
Jupyter Notebook
Untitled Last Checkpoint: 3 hours ago (autosaved) [Python 3]

Python 3

    File
    Edit
    View
    Insert
    Cell
    Kernel
    Widgets
    Help

# **Traffic Sign Recognition** 

​

## Writeup

​

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

​

---

​

**Build a Traffic Sign Recognition Project**

​

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)

* Explore, summarize and visualize the data set

* Design, train and test a model architecture

* Use the model to make predictions on new images

* Analyze the softmax probabilities of the new images

* Summarize the results with a written report

​

​

[//]: # (Image References)

​

[image1]: ./report-images/dataset-overview.jpg "Dataset Overview"

[image2]: ./report-images/example-distribution.jpg "Example Distribution"

[image3]: ./report-images/web-images.jpg "Web images"

​

​

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

​

---

### Writeup / README

​

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

​

You're reading it! and here is a link to my [project code](https://github.com/tachwali/SelfDrivingCars/tree/master/traffic_sign_classifier/Traffic_Sign_Classifier.ipynb)

​

### Data Set Summary & Exploration

​

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

​

I used the pandas library to calculate summary statistics of the traffic

signs data set:

​

* The size of training set is 34799

* The size of the validation set is 4410

* The size of test set is 12630

* The shape of a traffic sign image is 32x32x3

* The number of unique classes/labels in the data set is 43

​

#### 2. Include an exploratory visualization of the dataset.

​

Here is an exploratory visualization of the data set. 

​

![alt text][image1]

​

The distribution plots below show the number of examples per sign id is shown for training, validation and testing dataset:

​

![alt text][image2]

​

### Design and Test a Model Architecture

​

#### 1. Data Preprocessing

​

** Preprocessing. ** image data are normalized before being processed by subtracting the mean and dividing by the image standard deviation value 

​

​

#### 2. Classifier Architecture

​

Input shape is 32x32x3

​

**Layer 1: Convolutional.** 

​

Weight shape is 5x5x3x6 --> The output shape should be 28x28x6.

​

**Activation.** Relu activation function.

​

**Pooling.** Stride of 2 --> The output shape should be 14x14x6.

​

**Layer 2: Convolutional.** 

​

Weight shape is 5x5x6x16 --> The output shape should be 10x10x16.

​

**Activation.** Relu activation function.

​

**Pooling.** Stride of 2 --> The output shape should be 5x5x16.

​

**Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. 

5x5x16 --> 400

​

**Layer 3: Fully Connected.** 

This should have 400 inputs and 120 outputs.

​

**Activation.** Relu activation function.

​

**Layer 4: Fully Connected.** 

This should have 120 inputs aand 84 outputs.

​

**Activation.** Relu activation function.

​

**Layer 5: Fully Connected (Logits).** 

​

This should have 84 inputs and 10 outputs.

​

​

#### 3. Model Training

The following training parameters are used for the classifier model:

* Optimizer : AdamOptimizer 

* The batch size is 128 

* Number of epochs is 50 

* Learning rate is 0.001

​

To train the model, I used an cross-entropy as a cost function to be minimized. 

​

​

#### 4. Parameter Selection

​

My final model results were:

* training set accuracy of 1

* validation set accuracy of 0.972 

* test set accuracy of 0.94

​

An iterative approach was chosen for the parameter selection:

* The first architecture that was tried is the Lenet5 architecture which led an accuracy of .89. This architecture was chosen based on the course recomendation as a starting point for the classifier model. 

​

* I have added drop out to the model architecture to improve the validation accuracy since it is a method for regularizing the network and avoid overfitting. It seems that less keep_prob can be used than 0.9 since the accuracy of 1 in training results is still an indication of overfitting. However, I did not change the keep_drop value since the validation accuracy was acceptable.

​

* Without normalizing the input, the validation accuracy did not exceed 92%. The normalization has improved the performance of the network.

​

​

### Test a Model on New Images

​

#### 1. Examples of  German traffic signs found on the web 

​

Here are ten German traffic signs that I found on the web, I have used 10 images, however only 5 are shown below:

​

![alt text][image3] 

​

Note that the second image is harder to classify since it is dark. Despite the correct predictions observed by the classifier, this particular image had the least confidence compared to other prediction results. The preprocessing stage reduces the impact of brightness on the classification result. 

​

The model was able to correctly guess all the traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%

​

​

​

​

​


