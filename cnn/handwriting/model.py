from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

KERNEL_SIZE = (5, 5)


def run_it():
    # Define the CNN model
    """
    The model uses Keras Sequential API, which allows building neural networks by stacking layers linearly. This approach is suitable for most CNN architectures where data flows sequentially from input to output.
    :return:
    """
    model = models.Sequential()

    """
    This first convolutional layer performs several critical functions:
    
    32 filters: Creates 32 different feature maps to detect various patterns like edges, corners, and textures
    
    (5,5) kernel size: Uses 5×5 sliding windows to scan the input image, which is relatively large and can capture broader features
    
    ReLU activation: Introduces non-linearity by setting negative values to zero, helping the network learn complex patterns
    
    input_shape=(28, 28, 1): Specifies the input dimensions - 28×28 pixel grayscale images with 1 channel
    """
    model.add(layers.Conv2D(32, KERNEL_SIZE, activation='relu', input_shape=(28, 28, 1)))
    """
    The MaxPooling2D layer reduces spatial dimensions by taking the maximum value in each 2×2 region. This operation:
    
    Reduces computational complexity
    
    Provides translation invariance
    
    Helps prevent overfitting by reducing the number of parameters
    """
    model.add(layers.MaxPooling2D((2, 2)))

    """The second convolutional layer doubles the number of filters to 64, allowing the network to learn more complex, higher-level features. 
    The same 5×5 kernel size and ReLU activation are used, followed by another 2×2 max pooling operation."""
    model.add(layers.Conv2D(64, KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    """The Flatten layer converts the 2D feature maps into a 1D vector, preparing the data for the fully connected layer. 
    The Dense layer with 10 neurons corresponds to the 10 digit classes (0-9), using softmax activation to output probability distributions across all classes."""
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    # Split the data into training and test sets
    """The MNIST dataset contains 70,000 grayscale images of handwritten digits: 60,000 for training and 10,000 for testing. 
    Each image is 28×28 pixels with values ranging from 0 to 255."""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    """The reshaping operation adds a channel dimension, converting the shape from (60000, 28, 28) to (60000, 28, 28, 1). 
    Normalization divides pixel values by 255, scaling them to the range , which helps with training stability and convergence."""
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    """The to_categorical function converts integer labels (0-9) into one-hot encoded vectors. For example, the digit 3 becomes ."""
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Use the training data to train the model
    """The compilation step configures the training process:

    categorical_crossentropy: Loss function suitable for multi-class classification with one-hot encoded labels
    
    SGD optimizer: Stochastic Gradient Descent, a basic but effective optimization algorithm
    
    accuracy metric: Tracks the percentage of correct predictions during training"""
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamW', #sgd
                  metrics=['accuracy'])

    """The training process uses several key parameters:

    batch_size=100: Processes 100 samples at once before updating weights, balancing memory usage and gradient stability
    
    epochs=5: Passes through the entire training dataset 5 times
    
    verbose=1: Displays progress information during training"""
    model.fit(train_images, train_labels,
              batch_size=100,
              epochs=5,
              verbose=1)

    # Test the model's accuracy with the test data
    """The evaluation step tests the trained model on unseen data to assess its generalization performance."""
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

if __name__ == "__main__":
    # Example usage and parameter counting
    print(tf.__version__)         # Should output 2.9.1+
    print(tf.keras.__version__)   # Should match 2.9.0+
    run_it()