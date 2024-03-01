import os
from tensorflow import keras
import numpy as np

# This function loads an image from a given path and preprocesses it for prediction.
# The image is resized to 64x64 pixels, converted to an array, and a dimension is added to the array.
def load_test_image(path):
    test_image = keras.preprocessing.image.load_img(path, target_size=(64, 64))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    return test_image

# This function uses the loaded model to predict the class of the given image.
# The prediction result is printed, and if the result is 1, the image is classified as 'Real', otherwise, it's classified as 'AI'.
def predict(model, test_image):
    result = model.predict(test_image)
    print(result)
    if result[0][0] == 1:
        prediction = 'Real'
    else:
        prediction = 'AI'
    print(prediction)

# The directory of the current file is obtained.
file_dir = os.path.dirname(__file__)

# Paths for the model and test images are created.
model_path = os.path.join(file_dir, 'model_cnn.keras')
test_image1_path = os.path.join(file_dir, 'bird.png')
test_image2_path = os.path.join(file_dir, 'flower.png')
test_image3_path = os.path.join(file_dir, 'paint.jpg')

# The model is loaded.
loaded_model = keras.models.load_model(model_path)

# The test images are loaded and preprocessed.
image1 = load_test_image(test_image1_path)
image2 = load_test_image(test_image2_path)
image3 = load_test_image(test_image3_path)

# The model is used to predict the class of the test images.
predict(loaded_model, image1)
predict(loaded_model, image2)
predict(loaded_model, image3)
