# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
### Problem statement

The task involves using a Convolutional Neural Network (CNN) to classify images from the MNIST dataset, which contains hand-written digits. Moreover, noise has been added to the data, suggesting that this project may require techniques for denoising or managing noisy images.

### Dataset
![image](https://github.com/user-attachments/assets/197641e6-b6de-400a-b9ea-923effd18c95)


## Convolution Autoencoder Network Model
![image](https://github.com/user-attachments/assets/87de00e3-f647-4fda-989d-5185addcab11)


### STEP 1:
Data Loading: The MNIST dataset is loaded using mnist.load_data().
### STEP 2:
Data Preprocessing:
Training and test data are scaled to the range [0, 1].
The data is reshaped into 28x28 images with one channel (grayscale).
### STEP 3:
Data Augmentation:
Noise is added to the dataset with a noise factor to test how the model handles noisy input.
### STEP 4:
Design the model.

## PROGRAM
### Name:Suji G
### Register Number:212222230152

#### Import necessary packages
```

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
```

#### Load the Mnist Handwritten-dataset
```
(x_train, _), (x_test, _) = mnist.load_data()
```
#### Display the Data
```
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
#### View the shape of the dataset
```
x_train.shape
```
#### Scale the dataset and normalize them
```
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```

#### Display the images with noise
```
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

#### Create the model
```
input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(16,(3,3), activation = 'relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), padding = 'same')(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Encoder output dimension is ## Mention the dimention ##

# Write your decoder here
x = layers.Conv2D(8,(3,3), activation = 'relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(16,(3,3), activation='relu')(x)
x = layers.UpSampling2D((2,2))(x)
decoded_output=layers.Conv2D(1,(3,3), activation='sigmoid', padding='same')(x)

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)


print('Name:Suji G   Register Number: 212222230152      ')
autoencoder.summary()
```
#### Compile the model
```
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

#### Fit the model
```
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
```

#### Compare the original, noise, and de-noised images
```
n = 10
print('Name: Suji G    Register Number: 2122222230152')
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-11-11 172559](https://github.com/user-attachments/assets/f8157183-6206-4970-9778-9c005915ef41)


### Original vs Noisy Vs Reconstructed Image
![image](https://github.com/user-attachments/assets/ab454b3d-52b8-4550-9ee0-4bb279114ca8)




## RESULT
Thus, the program to incorporate auto encoder and decoder is implemented successfully for image denoising
