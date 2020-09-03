
# Real Time Digit Classifier
##### Summer Project 2019

<p align = "center">
 <img width = "460" height = "300" src = "https://github.com/IvLabs/Real-Time-Digit-Classifier/blob/master/Demos%20and%20Plots/Real%20Time%20Half%20Demo.gif">
   </p>
<h3 align = "center"><ins><b><u>Motivation</b></ins></h3>
<p align = "center">
To develop from scratch, a Real Time Digit Recognizer using elementary concepts of Deep Learning and Convolutional Neural Networks along with a  tinge of Image Processing and Object Tracking.
</p>
<br>

<h3 align = "center"><ins>Approach</ins></h3>


<b><ins>The Prelimnary Stage</b></ins> involved studying and learning the basics of Machine Learning and Deep Learning algorithms along with the understanding of the most primitive but effective Optimization algorithm which is the good ol' Gradient Descent and it's application to Single Node, Multi Node and even Multi Hidden Layer networks. <br><br>
<b><ins>Coding Networks from scratch</b></ins> using NumPy and writing all the functions for Forward and Backward passes along with activations and calculating gradients and putting it all into an iterative learning function which leanrs and returns the optimized values of the Parameters provided the Learning Rate, Number of Iterations, Batch Size, Number of Epochs and other HyperParameters, helps one deepen their understanding of the subject and experience the complications that arise when implementing models from scratch hence serving the purpose of this project. <br><br><br>

<h3 align = "center"><ins>Models and Frameworks</ins></h3> <br>

<b><ins>A Binary Classifier</b></ins> can be used as a basic project to gain intuitions and get things going. It was implemented using a Single Node archtecture and a Single Hidden Layer Architecture and classifies the selection of students to an institute depending on the marks in 2 exams from this [Data Set](./Data Sets/bindata.csv). <br><br>

<b><ins>Digit Classifier from Scratch</b></ins> is implemented using 2 approaches, code for both was written from scratch using basics of NumPy and was traied using the [MNIST Dataset](./Data Sets/mnist.pkl.gz). <br>
The First being a <b>Single Layer Perceptron</b> with only <b>10 Nodes using SoftMax Activation</b> and <b>Gradient Descent Optimization</b> and <b>Gaussian Initialization (Var = 1)</b> of weights.
<p align = "center">
 The image below shows the Learning Curves for the Single Layer Perceptron with <b>Learning Rate = 0.1</b>
 <img width = "460" height = "300" src = "https://github.com/IvLabs/Real-Time-Digit-Classifier/blob/master/Demos%20and%20Plots/SingleLayer.png">
   </p>
 
The Second is a <b>Multi Layer Perceptron</b> containing a single hidden layer with <b>200 Nodes</b> using <b>ReLU Activation</b> and an Output Layer with <b>10 Nodes</b> using <b>SoftMax Activation</b>. The optimization algorithm used was <b>Gradient Descent</b> and the weights were initialized using a <b>Gaussian Distribution (Var = 1)</b>. <br>
<p align = "center">
 The image below shows the Learning Curves for the Multi Layer Perceptron with <b>Learning Rate = 0.3</b>
 <img width = "460" height = "300" src = "https://github.com/IvLabs/Real-Time-Digit-Classifier/blob/master/Demos%20and%20Plots/MultiLayer.png">
   </p> <br><br>
   
<b><ins>The Convolutional Approach</b></ins> involves training a Convolutional Neural Network with <b>2 Convolutional Layers</b>, both using:<br>
Kernel Size = <b>5</b><br>
Padding = Same <b>(2)</b><br>
Stride = <b>1</b><br>
Followed by <b>Max Pooling</b> where,<br>
Kernel Size = <b>2</b><br>
Stride = <b>2</b><br>
I/O channels in Layer 1 : <b>1/32</b><br>
I/O channels in Layer 2 : <b>32/64</b><br>
And finally the <b>ReLU activation</b>.

This is then followed by a <b>Fully Connected Layer</b> with,<br>
Nodes = <b>1000</b><br>
Activation = <b>ReLU</b>

Finally, the Output Layer which implements <b>SoftMax Activation using 10 Nodes</b>.<br>
The optimizer used was <b>Adam Optimizer</b>.
The model was trained using PyTorch to speed up the training time. It used the MNIST Dataset as before. <br>
The trained model was then stored to a local directory.
<p align = "center">
 The image below shows the Learning Curves for the Convolutional Nueral Network with <br><b>Learning Rate = 0.001<br>Batch Size = 128<br>Number Of Epochs = 5</b><br>
 <img width = "460" height = "300" src = "https://github.com/IvLabs/Real-Time-Digit-Classifier/blob/master/Demos%20and%20Plots/CNN.png">
   </p> <br><br>

<b><ins>Digit Pad</b></ins> is a model developed to facilitate the user in drawing white digits onto a Black Drawing Pad of <b>128x128pi</b> by using the mouse events available in OpenCV and then resizing the image to <b>28x28pi</b> and passing it to the trained CNN model for classification.
<p align = "center">
 <img width = "600" height = "270" src = "https://github.com/IvLabs/Real-Time-Digit-Classifier/blob/master/Demos%20and%20Plots/Write%20Pad.gif">
   </p> <br><br>

<b><ins>Real Time Digit Classifier</b></ins> involves implementing the basics of Image Processing using OpenCV to capture and process the video input by applying <b>Thresholding using HSV Color Space</b> and <b>Centroid Tracking</b> to obtain the digit drawnn by the user. This is rescaled and drawn on a black and white image of suitable size (128x128pi) which is then resized to <b>28x28pi</b> and passed to the trained CNN model to obtain a prediction. <br>
*The Full Demo has been uploaded <a href = "https://github.com/IvLabs/Real-Time-Digit-Classifier/blob/master/Demos%20and%20Plots/Real%20Time%20Full%20Demo.mp4">here.</a>
<br><br><br>
<br>
References and aiding articles can be found in the <a href = "https://docs.google.com/document/d/1FpmhtFRAo3IJ94NWfQqjc38tlslT47FlybiaOEYPe3k/edit?usp=sharing">Documentation</a> provided. <br>Further specifications with results are available on the <a href = "https://www.ivlabs.in/mnist.html">Project Website</a>.*


