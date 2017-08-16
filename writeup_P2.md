#**Traffic Sign Recognition** 

## Kerem Par
<kerempar@gmail.com>




---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/visualization1.png =350x600 "Visualization of Data Set"
[image2]: ./writeup_images/sign_images.png =350x550 "Visualization of Sign Images"
[image3]: ./writeup_images/warping.png =300x250 "Warping"
[image4]: ./writeup_images/augmentation.png =350x600 "Augmentation"
[image5]: ./writeup_images/grayscaling.png =350x550 "Grayscaling"
[image6]: ./writeup_images/normalization.png =350x550 "Normalization"
[image7]: ./writeup_images/accuracy.png =350x500 "Validation and Training Accuracy"
[image8]: ./writeup_images/new_images.png =350x500 "New Images"
[image9]: ./writeup_images/probabilities.png =500x650 "Probabilities"
[image10]: ./writeup_images/probabilities2.png =500x650 "Probabilities"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kerempar/CarND-Term1-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart for each set (training, validation and testing) showing how the number of samples for each class are distributed. 

![alt text][image1]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale since many signs had similar color patterns and no advantage comes from using color except for some cases. It is reported that the grayscaling approach produced better results in the literature. It simplifies the 3 channels into a single channel, which can lead to quicker model training. Taking away the color channels might allow the model to focus on extracting the important non-color features better.

Here is a set of traffic sign images before and after grayscaling.

Before grayscaling:

![alt text][image2]

After grayscaling:

![alt text][image5]

As a last step, I normalized the image data by using the formula img = img / np.std(img) to get rid of brightness variations. I also applied the mean subtraction technique (img = img - np.mean(img)).

After normalization:

![alt text][image6]

I decided to generate additional data because some of the classes were represented far more than others, in some cases more than a factor of ten. This should be a problem for a machine learning algorithm, because the lack of balance in the data will lead it to become biased toward the classes with more samples. 

To add more data to the the data set, I determined a minimum number of samples which is 1500. I looped through each class (1 through 43), and if there were less than 1500 samples I then picked random images from that class and generated "warped" copies of them until there were 1500 samples total. The warping process consisted of applying a small but random vertical and horizontal shift (-2, 2 pixels), a small but random amount of rotation (-10, 10 degrees) by using OpenCV's affine transformation and a small amount of perspective modification (2 pixels) by using OpenCV's perspective transformation functions.   

Here are two examples of original images and augmented (warped) images:

![alt text][image3]

The difference between the original data set and the augmented data set is the following. The training set has a more balanced distribution. The number of examples in the training set was increased from 34799 to 67380.  

![alt text][image4]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet model architecture as the starting point. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6
| RELU					|						
| Max pooling	      	| 2x2 stride,  outputs 14x14x6
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16  
| RELU					|         			
| Max pooling	      	| 2x2 stride,  outputs 5x5x6
| Flatten		       	| outputs 400			
| Fully connected		| outputs 120
| RELU					|         			
| Dropout				| keep probability 0.5
| Fully connected		| outputs 84
| RELU					|         			
| Dropout				| keep probability 0.5         
| Fully connected (Logits)		| outputs 43									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a learning rate of 0.00097 (I started with 0.001, then I just tried a slightly different learning rate to see the effect, but I could not see much difference though). Number of EPOCS was 25 . I used a batch size of 128.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.986
* validation set accuracy of 0.953
* test set accuracy of 0.933

The following figures show how training and validation accuracy changes through the iterations. 

![alt text][image7]

I used the LeNet model architecture as the starting point with the implementation used during the course. It was already proved to be very successfull for a similar image recognition problem. It is an example of convolutional network that can automatically learn hierarcies of features. 

Initially I could get a validation accuracy of 0.90-0.91 (with only preprocessing) which is less than required level. I did not make major changes on the architecture. After augmentation of the data set, I got a small increase to 0.92 level. I decided to add "dropout" which was discussed to have major impacts on the performance during the course I used a keep probability of 0.5. This significantly increased the validation accuracy to nearly 0.95-0.96 which satisfies the requirement. I tried also 0.6 as the keep probability, I could not observe noticeable changes. 

One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The final training set and validation set accuracies can be further improved.  

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image8] 

Actually, the quality of the images are high, the features can be clearly seen. Normally, a high recognition rate can be expected. However, the first image might be difficult to classify because there are many similar sign classes like 20, 80 km/h restrictions. On the other hand, I observed that the quality of image examples in the training set are not so high and the network might have been trained more biased to similar low quality images. That's why, it might be possible to expect some surprising results with high quality images.   

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

I have trained the network and run predictions several times. I observed different results. Once the model was able to correctly guess all of the traffic signs (accuracy of 100%). Another run predicted 5 of the 8 traffic signs, which gives an accuracy of 67,5%. Here are the results of the two predictions:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30 km/h)       		| Speed limit (30 km/h)   									| 
| Go Straight or left    			| Go Straight or left  	
| Ahead only    			| Ahead only		
| Bumpy road    			| Bumpy road										|
| General caution					| General caution										|
| No entry	      		| No entry					 				|
| No vehicles			| No vehicles     		
| Turn right ahead					| Turn right ahead


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30 km/h)       		| Speed limit (20 km/h)   									| 
| Go Straight or left    			| Go Straight or left  	
| Ahead only    			| Ahead only		
| Bumpy road    			| Bumpy road										|
| General caution					| General caution										|
| No entry	      		| No entry					 				|
| No vehicles			| No vehicles     		
| Turn right ahead					| Stop


The accuracy of first run is higher than the accuracy on the test test which is 0.933. The accuracy of second run is lower than the accuracy on the test set. Most of the time, I observed that the accuracy was slightly lower than the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

Here are the top five soft max probabilities for each image along with the sign type of each probability for the two runs.

First run (accuracy of 100%):

![alt text][image9] 

Second run (accuracy of 62.5%):

![alt text][image10] 

In general, I observed that the probabilities are close to each other. For instance, for the first image, during first run the model guesses the correct sign (30 km/h) with a probability of 16% compared to 15% of the second guess which is 50 km/h. During second run, the probabilites of the first and second guesses are nearly the same, and the model misses the correct one and guesses 20 km/h instead of 30 km/h. 	


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


