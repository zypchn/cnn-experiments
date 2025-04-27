# cnn-experiments

<br/>

## Introduction
Convolutional Neural Network (CNN) is an advanced version of artificial neural networks (ANNs), primarily designed to extract features from grid-like matrix datasets. This is particularly useful for visual datasets such as images or videos, where data patterns play a crucial role. CNNs consist of multiple layers like the input layer, Convolutional layer, pooling layer, and fully connected layers. <br/>

![image](https://github.com/user-attachments/assets/95494af0-1b30-4992-8e62-8166e493aa18)

- **Input Layer:** It’s the layer in which we give input to our model. In CNN, Generally, the input will be an image or a sequence of images. This layer holds the raw input of the image with width 32, height 32, and depth 3. 
- **Convolutional Layer:** This is the layer, which is used to extract the feature from the input dataset. It applies a set of learnable filters known as the kernels to the input images. The filters/kernels are smaller matrices usually 2×2, 3×3, or 5×5 shape. it slides over the input image data and computes the dot product between kernel weight and the corresponding input image patch. The output of this layer is referred as feature maps.
- **Activation Layer:** By adding an activation function to the output of the preceding layer, activation layers add nonlinearity to the network. It will apply an element-wise activation function to the output of the convolution layer. Some common activation functions are RELU, Tanh, Leaky RELU, etc. <br/>
- **Pooling Layer:** This layer is periodically inserted in the covnets and its main function is to reduce the size of volume which makes the computation fast reduces memory and also prevents overfitting. Two common types of pooling layers are max pooling and average pooling. <br/>
- **Flattening:** The resulting feature maps are flattened into a one-dimensional vector after the convolution and pooling layers so they can be passed into a completely linked layer for categorization or regression.
- **Fully Connected Layers:** It takes the input from the previous layer and computes the final classification or regression task. <br/>

The purpuse of this study was to conduct a comparative analysis to demonstrate the capabilities of different CNN architectures. 

<br/>

## Methods
The first step of the study was to find a benchmark dataset that can be quickly trained. I choose [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) from *torchvision* library. The dataset consists a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes: 
*T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot*.

Then, I defined 4 models with the following architectures:
**1)** Basic CNN with 3 Convolutional Layers, 2 Max Pooling Layers and 1 Fully Connected Layer
**2)** Same as the 1st architecture, but with some additional layers like dropout, batch_norm, etc.
**3)** Pretrained EfficientNet_B0 
**4)** Feature extraction was done using CNN, the features were then fed into a xxx model.

For implementation, I used the following libraries:
- **numpy:**
- **torch:**
- **timm:**
- **sklearn:**
- **matplotlib:**
- **tqdm:**
