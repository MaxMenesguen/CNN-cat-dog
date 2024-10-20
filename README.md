# Cat vs Dog Image Recognition

This project is a convolutional neural network (CNN) built from scratch using PyTorch to classify images as either cats or dogs. The project demonstrates the implementation of a CNN for binary image classification, with a simple user interface to upload images and display predictions.

## Features
- **Cat/Dog Classification**: Classifies images into two categories (cat or dog) with high accuracy.
- **Convolutional Neural Network**: Uses multiple convolutional, pooling, and fully connected layers to process images.
- **User Interface**: Provides an interface where users can upload images and see real-time classification results.
- **Data Preprocessing**: The images are transformed (resized, grayscaled, and normalized) before being passed to the network.

## Technologies Used
- **Python**
- **PyTorch**: For building and training the convolutional neural network.
- **HTML/CSS**: For the UI layout.

## Model Architecture
The CNN is designed with:
- **Convolutional Layers**: To extract feature maps from the images.
- **Max Pooling Layers**: To reduce the spatial dimensions.
- **Fully Connected Layers**: To classify the extracted features into the cat or dog category.

The model was trained on a dataset of cat and dog images, and its performance was evaluated using standard accuracy metrics.

## Installation

### Prerequisites
- Python 3.x
- PyTorch
- Node

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MaxMenesguen/CNN-cat-dog.git
   cd CNN-cat-dog
   ```
2. **Install the Dependencies**:
   ```bash
   pip install PyTorch Node
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the UI**:
   - Open a browser and navigate to `http://localhost:5000`.
   - Upload an image (cat or dog) and press the "Recognize" button to get the classification result.

## How It Works
1. **Image Input**: The user uploads an image through the web interface.
2. **Image Preprocessing**: The image is resized, converted to grayscale, and normalized to match the input requirements of the neural network.
3. **Prediction**: The preprocessed image is passed through the CNN, which outputs a classification as either a cat or a dog.
4. **Output**: The result is displayed on the interface along with the model's confidence.

## Example Output
- **Uploaded Image**: Displays the uploaded image.
- **Transformed Image**: Shows the grayscale, resized version of the image that is fed to the neural network.
- **Classification Result**: Displays the prediction ("It's a cat!" or "It's a dog!") and the confidence score.

## Dataset
The model was trained on a subset of the [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data), which contains 25,000 labeled images of cats and dogs.
