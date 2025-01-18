# Overview 
Built a neural network from scratch using only NumPy in Python, achieving a 95.45% accuracy in recognizing handwritten digits.

# Vision 
Humanity has evolved over the course of long history, taking countless years to reach where we are today. Throughout this long journey, there have been events that accelerated human progress at extraordinary speeds—such as the utilization of fire, the development of language, the Industrial Revolution, the invention of electricity, and the spread of the internet. Now, is the evolution of AI one of those events? If so, being born in an era where such a transformative event is happening fills me with excitement. At the same time, I feel that not pursuing this field would mean missing out on something significant. As I thought about this, I began to grow curious about AI—how neural networks work, how they learn, and how they "think." This curiosity led me to start studying neural networks by reading Michael Nielsen's online book, [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/).

As I read through the book, Michael teaches how neural networks work by explaining the underlying mathematics and shows how to implement one. After I implemented all the neural network functionalities, such as feedforward, backpropagation, and stochastic gradient descent.

# Performance
I trained the network on 70,000 images of handwritten digits, and after 30 epochs of training, the network achieved an accuracy of 95.16%. The image below shows the accuracy of the neural network after each epoch. 

<img width="178" alt="Image" src="https://github.com/user-attachments/assets/0c049289-f4c2-45dc-a5e9-a7f4eaa309b0" />   <img width="176" alt="Image" src="https://github.com/user-attachments/assets/0744bc64-72c1-4166-b4c8-495e80b4a23b" />

Performing another stochastic gradient descent with a lower eta will improve the network's accuracy to 95.45%.

# Application
As I continued reading through the book, Michael mentioned, 'Once we've learned a good set of weights and biases for a network, it can easily be ported to run in JavaScript in a web browser.' This piqued my curiosity, so I decided to try it. Below is a part of the website called Mosaicify that I made, where I have ported the neural network I trained to run in TypeScript.
<div style="text-align: center;">
<img width="296" alt="Image" src="https://github.com/user-attachments/assets/70b5b880-c936-459b-8795-de5a0d81f3dc" />
</div>
Although the success rate has decreased due to training-serving skew (the mismatch between training data and actual input), the rate is still decent as I implemented centering, scaling, and downsampling algorithms to make the input more similar to the training data.
