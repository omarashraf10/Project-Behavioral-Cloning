# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report





## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 the recorded video of the atonums mode of my model

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

i used more powerfull network produced by nividia team which is illustrated in this lesson in point 15 ,
but with some changes to reduce overfitting and better accuracy like adding many droput layers .

The model includes RELU layers to introduce nonlinearity after each convulotion layer (code line 82,84,86,87), and the data is normalized in the model using a Keras lambda layer (code line 80 ). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 83,85,88,91 ) with prpability 0.5 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 9-57). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

i collected large amount of data from training mode of the simulator and trained it till i reached to the final data which are straight data in the middle of the rad which in the folder imgs and recovery data which in folders tricky_imgs and trickys_imgs.

all the data i collected is in this repo :

https://github.com/omarashraf10/my-collected-data

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ..

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to get perfect accuracy in training and validation set and drive the car safely in atonumos mode

My first step was to use a convolution neural network model similar to the nividia architecture I thought this model might be appropriate because it is a powerfull model for this topic 

i used  a lambda layer is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py (line 80)

then i uesd data augmentation which is A effective technique for helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement. (line 61-66)

then i cropped images by adding the cropping layer, the model will automatically crop the input images when making predictions in drive.py.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that i added four droupout layers ,one after each convolution layer and one after a dense layer .


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I recollect more data of the road which the car is in an edge then back to the middle

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 79-94) consisted of a convolution neural network with the following layers and layer sizes ...
the model :-

-a lambda layer
-a cropping layer with cropping=((70,25),(0,0))
-a convolution layer which dimensions are 24,5,5 
-activation layer "relu"
-a droupout layer with propability 0.5
-a convolution layer which dimensions are 36,5,5 
-activation layer "relu"
-a droupout layer with propability 0.5
-a convolution layer which dimensions are 48,5,5 
-activation layer "relu"
-a convolution layer which dimensions are 64,5,5 
-activation layer "relu"
-a droupout layer with propability 0.5
-a flatten layer
-a dense layer with size 100
-a droupout layer with propability 0.5
-a dense layer with size 10
-a dense layer with size 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

[image5]: ./examples/center_2019_02_14_12_50_51_658.jpg 

![alt text][image5]

[image6]: ./examples/center_2019_02_14_12_51_07_111.jpg

![alt text][image6]



I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to the middle of the road These images show what a recovery looks like  :

[image2]: ./examples/center_2019_02_14_12_26_06_021.jpg

![alt text][image2]

[image3]: ./examples/center_2019_02_14_12_26_06_948.jpg

![alt text][image3]

[image4]: ./examples/center_2019_02_14_12_26_04_632.jpg

![alt text][image4]


[image1]: ./examples/center_2019_02_14_11_28_44_176.jpg "center Image"

![alt text][image1]


To augment the data sat, I also flipped images and angles thinking that this would help the model to generalize better and  give more better accuracy .
the code used for augmentation in lines (61 to 66)

flipping the image by the function :cv2.flip(image,1)
and the measurment of the flipped image was the negative of the measurment of the original image


After the collection process, I had X number of data points. I then preprocessed this data by using cropping technique by using cropping2d function to cut off not useful areas and i used lambda layer to normalize the data as in lines (80 and 81)


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the validation error is minimum after 3 epochs I used an adam optimizer so that manually training the learning rate wasn't necessary.
