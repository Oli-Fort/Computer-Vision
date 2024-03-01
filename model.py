from tensorflow import keras
import pickle
import os

# This function loads data from a given path.
# The specifics of the data loading process are not shown in this excerpt.

# This function creates a Sequential CNN model with Keras.
# The model has a Conv2D layer, a MaxPool2D layer, a Flatten layer, and two Dense layers.
# The model is compiled with binary crossentropy loss, Adam optimizer, and accuracy as the metric.
def load_data(path):
    data = keras.utils.image_dataset_from_directory(path, batch_size=32, image_size= (64, 64))

    return data

# This function trains the CNN model with the provided training and test sets for 10 epochs.
def make_model():
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, input_shape= [64, 64, 3], activation='relu'))
    cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(units=128, activation='relu'))
    cnn.add(keras.layers.Dense(units=1, activation='sigmoid'))
    cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return cnn

# This function saves the trained model to a specified file path.
# If the directory for the file path does not exist, it creates it.
def train_model(cnn, training_set, test_set):
    cnn.fit(x= training_set, validation_data= test_set, epochs=10)
    return cnn

def save_model(mdl, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    mdl.save(file_path)

# The directory of the current file is obtained.
file_dir = os.path.dirname(__file__)

# Paths for the training set, test set, and model are created
training_set_path = os.path.join(file_dir, 'train_set')
test_set_path = os.path.join(file_dir, 'test_set')
model_path = os.path.join(file_dir, 'model_cnn.keras')

# The training and test sets are loaded.
training_set = load_data(training_set_path)
test_set = load_data(test_set_path)

# A CNN model is created.
model = make_model()

# The model is trained with the training and test sets.
cnn = train_model(model, training_set, test_set)

# The trained model is saved.
save_model(cnn, model_path)
