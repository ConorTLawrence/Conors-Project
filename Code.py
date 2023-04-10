import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Define the list of kitchen knife classes
classes = ['Boning Knife', 'Carving Knife', 'Butcher Knife', 'Cleaver', 'Chefs Knife', 'Santoku', 'Stapula Knife']

# Load the input image and preprocess it
img_path = 'path/to/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Use the model to predict the class probabilities for the input image
preds = model.predict(x)

# Decode the predicted class probabilities into class names
results = decode_predictions(preds, top=1)[0]

# Print the predicted class name
predicted_class = results[0][1]
if predicted_class in classes:
    print(f'The predicted kitchen knife is: {predicted_class}')
else:
    print('The input image is not a kitchen knife.')
