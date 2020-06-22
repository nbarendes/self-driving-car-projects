# **Traffic Sign Recognition** 



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"



---


### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![exemple](https://user-images.githubusercontent.com/34095574/78636417-95209200-78a8-11ea-838a-3458d0bfed8d.jpg)

This bar chart is showing how many images there are in the traninig set for each label.


![graph](https://user-images.githubusercontent.com/34095574/78636909-969e8a00-78a9-11ea-9dc5-be9010344591.jpg)

### Design and Test a Model Architecture

#### 1. Data preprocessing

1) each image is converted from RGB to YUV color space, then only the Y channel is used.
2) contrast of each image is adjusted by means of histogram equalization. This is to mitigate the numerous situation in which the image contrast is really poor.
3) each image is centered on zero mean and divided for its standard deviation. This feature scaling is known to have beneficial effects on the gradient descent performed by the optimizer.

Here is an example of a traffic sign image  after preprocessing.

![preprocess](https://user-images.githubusercontent.com/34095574/78638218-32c99080-78ac-11ea-8809-cf5b22721a46.jpg)


To get additional data, I leveraged on the ImageDataGenerator class provided in the Keras library. Training images are randomly rotated, zoomed , flipped, and shifted but just in a narrow range, in order to create some variety in the data while not completely twisting the original feature content. 
Here is an example of an original image and an augmented image:

![im](https://user-images.githubusercontent.com/34095574/78637548-d9ad2d00-78aa-11ea-88ee-6065a70dc2d5.jpg)




#### 2. Design and Test a Model Architecture 

Here I used a self defined model(TrafficNet). 
My model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1  image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Dropout               |                                               |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 12x12x64    |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 6x6x128 				    |
| Dropout               |                                               |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 4x4x64      |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 2x2x64 				    |
| Dropout               |                                               |
| flatten               | outputs 17408                                 |
| Fully connected		| inputs 17408, outputs 64        				|
| Fully connected		| inputs 64, outputs 43        					|
| Softmax				|           									|

 


#### 3. Description of my trained model.

To train the model, I used Stochastic Gradient Descent optimized by AdamOptimizer at a learning rate of 0.001. Batchsize was set to 128 due to memory constraint and number of epochs equaled 25.

The approach to classify the traffic symbols was to implement my own model(TrafficNet) and iteratively tune it to improve performance for this specific dataset. The trafficnet model comprises of a stack of three convolution layers and two fully connected layers with RELU activations interleaved betweeen them. The convolutions layers outputs are also fed through MaxPooling layers after RELU. One of the changes that improved performance for this dataset is the inclusion of dropout layers. This was added when I noticed the model was overfitting to the training data set. Learning rate, batch size and the probablity for the dropout layers were the most important hyperparameters that I had to tune. My initial learning rate of 0.1 with the GradientDescent optimizer was failing to train, possibly getting stuck at a local optima. Reducing learning rate by an order was sufficient to get the model to train. I also switched the optimizer to AdamOptimizer as it converged significantly faster than GradientDescent.


My final model results were:
* training set accuracy of 99.303%
* validation set accuracy of 99.181% 
* test set accuracy of 93.230%


 

### Test a Model on New Images

#### 1. Five German traffic signs found on the web.

All the 5 images are taken from real-world videos, thus they're far from being perfectly "clean". For example, in figure 2 (slippery sign) the image contrast and brightness is bad and might be difficult to classify.

Here are five German traffic signs that I found on the web:

![panneaux](https://user-images.githubusercontent.com/34095574/78644636-79bc8380-78b6-11ea-91c8-29e58f176662.jpg)



#### 2. Model's predictions on these new traffic signs.

Here are the results of the prediction:


![prediction](https://user-images.githubusercontent.com/34095574/78645794-572b6a00-78b8-11ea-9489-9ea09e326a3a.jpg)



The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.230%.

#### 3. top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Road work (probability of 1.00), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Road work  									| 
| 0.00     				| Bumpy road 									|
| 0.00					| Children crossing								|
| 0.00	      			| No passing vehicles over 3.5 metric tons		|
| 0.00				    | Go straight or left      						|

![1](https://user-images.githubusercontent.com/34095574/78648473-4e3c9780-78bc-11ea-9625-55f6656e8a20.jpg)


For the second image , the model is relatively sure that this is a Slippery road  (probability of 1.00), and the image does contain a Slippery road . The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Slippery road  								| 
| 0.00     				| Double curve 									|
| 0.00					| Wild animals crossing							|
| 0.00	      			| Dangerous curve to teh left		            |
| 0.00				    | Bicycles crossing      						|



![2](https://user-images.githubusercontent.com/34095574/78648475-4ed52e00-78bc-11ea-8ee2-3eaaeecd47a7.jpg)


For the third image , the model is relatively sure that this is a Speed limit(30km/h) (probability of 1.00), and the image does contain a Speed limit(30km/h) . The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit(30km/h)  							| 
| 0.00     				| Speed limit(50km/h)  							|
| 0.00					| Speed limit(20km/h) 							|
| 0.00	      			| Speed limit(80km/h) 		                    |
| 0.00				    | Speed limit(70km/h)       					|

![3](https://user-images.githubusercontent.com/34095574/78648476-4ed52e00-78bc-11ea-9632-16f425722352.jpg)



For the fourth image , the model is relatively sure that this is a General caution (probability of 1.00), and the image does contain a General caution . The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| General caution 							    | 
| 0.00     				| Traffic signals  							    |
| 0.00					| Bumpy road 							        |
| 0.00	      			| Road narrows on the right 		            |
| 0.00				    | Pedestrians       					        |

![4](https://user-images.githubusercontent.com/34095574/78648477-4f6dc480-78bc-11ea-97f5-38aa06316034.jpg)




For the fifth image , the model is relatively sure that this is a Bumpy road (probability of 1.00), and the image does contain a Bumpy road . The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bumpy road							        |    
| 0.00     				| Road work  							        |
| 0.00					| General caution 							    |
| 0.00	      			| Dangerous curve to the left 		            |
| 0.00				    | Traffic signals       					    |


![5](https://user-images.githubusercontent.com/34095574/78648471-4da40100-78bc-11ea-80fc-0ad12177939c.jpg)




#### 1. Visual output of your trained network's feature maps. 


### First convolutional layer


![image](https://user-images.githubusercontent.com/34095574/78664927-489f7b80-78d5-11ea-8536-4caddc9d9959.png)


### Second convolutional layer 
![image](https://user-images.githubusercontent.com/34095574/78664973-5ce37880-78d5-11ea-893c-eb55f449f8bf.png)

### Third convolutional layer

![image](https://user-images.githubusercontent.com/34095574/78665004-6bca2b00-78d5-11ea-9b30-1e898f4ad8c1.png)
