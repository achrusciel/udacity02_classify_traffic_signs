# **Traffic Sign Recognition** 

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

[image1]: ./writeup_images/exploratory_visualization.png "Visualization"
[image2]: ./writeup_images/grayscale_image.png "Grayscaling"
[image3]: ./writeup_images/data_augmentation.png "Data Augmentation"
[image4]: ./New_Images/1.jpg "Traffic Sign 1"
[image5]: ./New_Images/2.jpg "Traffic Sign 2"
[image6]: ./New_Images/3.jpg "Traffic Sign 3"
[image7]: ./New_Images/4.jpg "Traffic Sign 4"
[image8]: ./New_Images/5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! And here is a link to my [project code](https://github.com/achrusciel/udacity02_classify_traffic_signs/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas, numpy, and python set-data structures to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data:

![exploratory visualization][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

1) Generation of more data. The training set came with 34799. To make it more robust I duplicated every image in the training set twice. The first copy was rotated left, the second copy was rotated right. I therefore expanded the training set to 104397. Here is an example of an image in its original state and its two rotated copies:

![alt text][image3]

2) Apply grayscaling and increase of contrast. According to [Sermanet & LeCun, 2011] a higher accuracy can be achieved by converting images to grey. Here is an example of a traffic sign image before and after grayscaling and increase of contrast:

![alt text][image2]

3) Normalize image data: In this project I followed the recommendation of applying the following normalization: (pixel - 128)/ 128. Normalization is required because gradient descent algorithms perform better on normalized data.

| Example before normalization	|Example after normalization						| 
|:-----------------------------:|:-------------------------------------------------:| 
| array([[ 65],[ 35],[ 35], ...	| array([[-0.4921875],[-0.7265625],[-0.7265625], ...|

4) Gaussian Blur. I applied Gaussian blur, but this worsened my validation accuracy. Therefore I ommitted this step during paramter tuning.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grey image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					| Activation									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| Flatten output								|
| Fully connected		| Fully Connected. Input = 400. Output = 120	|
| RELU					| Activation									|
| Dropout				| Regularization								|
| Fully connected		| Fully Connected. Input = 120. Output = 84		|
| RELU					| Activation									|
| Fully connected		| Fully Connected. Input = 84. Output = 43		|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

1) Optimizer
	* AdamOptimizer: The AdamOptimizer generated high accuracy from the very beginning. In the initial LeNet model a validation accuracy of ~89% could be achieved. After applying the preprocessing steps, especially the grayscaling and generation of additional training data the validation accuracy of ~95% could be achieved. 
	* GradientDescentOptimizer: The GradientDescentOptimizer started with a very poor validation accuracy but increased steadily up to 90%, but not more.
	* Conclusion: I decided to continue with the AdamOptimizer
2) Batch Size
	* I left the batch size as it is. On my notebook with no gpu and 8 GB of memory the model trained for about 15 minutes and the memory utilization was up to 85%
	* Conclusion: this resource utilization seemed to be a good tradeoff for me.
3) Number of Epochs
	* Especially during my tests of the GradientDescentOptimizer I increased this value, which resulted in longer training phases of the model. But I could not get higher than ~90% validation accuracy. For the AdamOptimizer I saw that after the 7th epoch the validation accuracy stagnated at about ~95%.
	* Conclusion: I left the value 10. After rerunning the model several times it reached high enough validation accuracy.
4) Learning Rate
	* For the GradientDescentOptimizer the best learning rate I could figure out was 0.02. For the AdamOptimizer the best learning rate I found was 0.005
	* Conclusion: I chose the learning rate to be 0.005.
5) Dropout
	* I set the dropout rate to be at 0.3. Changes in this value did not lead to an increased validation accuracy.
	* Conclusion: I believe it plays more of a role on where in the architecture the dropout is applied.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training Accuracy = 0.978
* Validation Accuracy = 0.964
* Test Accuracy = 0.932

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
	* My initial architecture was the standard LeNet Convolutional network. I chose it as a starting point. 
* What were some problems with the initial architecture?
	*  No problems in particular, but the validation accuracy was only at 0.89%.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	* I introduced dropout after the first fully connected layer. Without dropout my training accuracy was at ~98%, but my validation accuracy was at ~94%. With the introduction of dropout the validation accuracy increased.
* Which parameters were tuned? How were they adjusted and why?
	* The Learning Rate of the AdamOptimizer was adjusted to 0.005. This valued performed best with respect to validation accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	* I majorly emphasized on the preprocessing of the data. The highest gains could be achieved by creating the training data sets (rotated copies of the existing training data). Another important step was to read the paper by [Sermanet & LeCun, 2011] and to preprocess the data (greyscaling, increasing contrast). This alone was enough to meet the requirement of achieving a validation accuracy of >93%. I realized that my model overfitted the training data, therefore I introduced dropout. The best result could be achieved with keep probability at 0.5.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

1) Stop Sign
	* The stop sign is clear and the contrast is good
	* Conclusion: Should be easy to classify
2) Speed limit to 100km/h
	* The speed limit sign is clear and there is little noise in the background of the sign.
	* During the scaling to fit the model's requirements of 32x32 pixels the information within the sign could be lost. That means that the value "100" could be blurred.
	* Conclusion: It could be hard to distinguish between potential other numbers.
3) General Caution
	* Below the sign is an additional sign with text. Behind the sign are trees and a building. That is a lot of noise.
	* Conclusion: Because of the noise it might be harder to classify.
4) Yield
	* The sign is clear, there exist no numbers or characters within the sign. Background noise is low.
	* Conclusion: should be easy to classify
5) Double curve
	* The borders of the triangle are partially covered in snow. This leads to a lower contrast between the border and the white space. As in the speed limit sign also the double curve image in the middle of the sign might get blurred when the image is rescaled into 32x32 pixels.
	* Conclusion: Could be hard to classify

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| Correct?	|
|:---------------------:|:---------------------------------------------:| :--------:|
| Stop Sign      		| Stop sign   									| Yes		|
| 100 km/h    			| Turn right ahead								| No		|
| General Caution		| Road work										| No		|
| Yield		      		| Yield				 							| Yes		|
| Double Curve			| No passing for vehicles over 3.5 metric tons	| Yes		|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 60%. This is worse than the expected accuracy as the test set might indicate.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

1) Stop Sign
For the first image, the model is sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop sign   									| 
| 0.0     				| 20 km/h	  									|
| 0.0					| 30 km/h										|
| 0.0	      			| 50 km/h					 					|
| 0.0				    | 60 km/h   	  								|


2) 100 km/h
For the second image, the model is sure that this is a turn right ahead sign (probability of 1.0), and the image does contain a speed limit of 100km/h. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.32442307e-01		| Turn right ahead								| 
| 6.75577521e-02		| Keep left	  									|
| 8.12942602e-10		| Go straight or left							|
| 2.39857734e-16		| 50 km/h					 					|
| 2.35281806e-19	    | 30 km/h   	  								|

3) General Caution
For the second image, the model is sure that this is a road work sign (probability of 1.0), and the image is general caution. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work 									| 
| 0.0     				| General Caution								|
| 0.0					| 20 km/h										|
| 0.0	      			| 30 km/h					 					|
| 0.0				    | 50 km/h   	  								|

4) Yield
For the second image, the model is sure that this is a yield (probability of 1.0), and the image is a yield sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield		 									| 
| 0.0     				| 20 km/h	  									|
| 0.0					| 30 km/h										|
| 0.0	      			| 50 km/h					 					|
| 0.0				    | 60 km/h   	  								|

5) Double Curve
For the second image, the model is sure that this is Double Curve sign (probability of 1.0), and the image is a double curve sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Double Curve 									| 
| 0.0					| 20 km/h										|
| 0.0					| 30 km/h										|
| 0.0	      			| 50 km/h					 					|
| 0.0				    | 60 km/h   	  								|

##### Conclusion
Looking at the softmax probabilities I can see that the speed limit signs are relativly often represented in the top 5 predictions. Looking at the distribution of the training data I can see that those signs are more often represented in the data set than other signs. Maybe this leads to an overall higher prediction probability for those signs. Especially in the case of the 100km/h speed limit sign I imagine that the classification error could be reduced.