# ==================================================
# ST1516 DEVOPS AND AUTOMATION FOR AI CA2 ASSIGNMENT
# ==================================================
# NAME: EDWARD TAN YUAN CHONG
# CLASS: DAAA/FT/2B/04
# ADM NO: 2214407
# ==================================================
# FILENAME: test_docker.py
# ==================================================

# Import modules
import pytest
import requests
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import test data
# The two different image sizes
image_size_128 = (128,128)
image_size_31 = (31,31)

def load_all_images(image_size, dataset_type):
    # Normalization
    datagen = ImageDataGenerator(rescale=1./255) 
    # Load image from directory
    generator = datagen.flow_from_directory(
        f"Dataset for CA1 part A/{dataset_type}",  
        target_size=image_size,
        color_mode='grayscale',
        batch_size=64, 
        class_mode='binary',
        shuffle=False
    )
    # List to store all the images and labels
    all_images = []
    all_labels = []
    # Calculate the number of loops needed to load all images and labels
    num_batches = len(generator)
    # Loop the number of loops needed and append each batch to the list
    for i in range(num_batches):
        batch_images, batch_labels = next(generator)
        all_images.extend(batch_images)
        all_labels.extend(batch_labels)
    # Return NumPy arrays
    return np.array(all_images), np.array(all_labels)

# Load datasets for 128x128 image size
X_test_128, y_test_128 = load_all_images(image_size_128, 'test')
# Load datasets for 31x31 image size
X_test_31, y_test_31 = load_all_images(image_size_31, 'test')

# Print length of each train, test, validation dataset
print(f"\n\nImage size: {image_size_128}")
print(f'Length of test array: {len(X_test_128)}.\n\n')
print(f'Length of test_label array: {len(y_test_128)}\n')
print("-"*50)
print(f"\n\nImage size: {image_size_31}")
print(f'Length of test array: {len(X_test_31)}.\n\n')
print(f'Length of test_label array: {len(y_test_31)}.')

# Server URL
server_url_31 = "http://ca2_models_serving:8501/v1/models/customvgg31:predict"
server_url_128 = "http://ca2_models_serving:8501/v1/models/conv2d128:predict"


def make_prediction_31(instances):
    data = json.dumps({"signature_name": "serving_default","instances": instances.tolist()}) 
    headers = {"content-type": "application/json"}
    json_response = requests.post(server_url_31, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions

def make_prediction_128(instances):
    data = json.dumps({"signature_name": "serving_default","instances": instances.tolist()}) 
    headers = {"content-type": "application/json"}
    json_response = requests.post(server_url_128, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions

@pytest.mark.parametrize('indexes',[[0,4],[50,54],[133,137]])
def test_prediction_31(indexes):
    predictions = make_prediction_31(X_test_31[indexes[0]:indexes[1]]) 
    for i, pred in enumerate(predictions):
        assert y_test_31[i] == np.argmax(pred)

@pytest.mark.parametrize('indexes',[[0,4],[15,19],[192,196]])
def test_prediction_128(indexes):
    predictions = make_prediction_128(X_test_128[indexes[0]:indexes[1]]) 
    for i, pred in enumerate(predictions):
        print(y_test_128[i] == np.argmax(pred))
        assert y_test_128[i] == np.argmax(pred)
