'''
Created on July 10th, 2018
Author: Julian Weisbord
Description: Reload the clothing classifier for inference.
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'
import numpy as np
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
import h5py

TEST_DATA = "./test_data/"
IMG_WIDTH = 300
IMG_HEIGHT = 300
BATCH_SIZE = 10
TEST_SAMPLES = 100
CLASSES = ["Jeans", "Sweatpants", "Blazer"]

def predict():
    '''
    Description: Print a prediction for all of the images in TEST_DATA
    '''
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
                        TEST_DATA,
                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                        classes=CLASSES,
                        batch_size=BATCH_SIZE)

    json_file = open("models/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/basic_cnn.h5")

    # prediction = loaded_model.predict_generator(test_generator, steps=1)
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    predictions = loaded_model.predict_generator(test_generator, steps=TEST_SAMPLES // BATCH_SIZE, verbose=0)
    prediction_accuracy = loaded_model.evaluate_generator(test_generator, TEST_SAMPLES)
    print(predictions)

    for prediction in predictions:
        cls_index = np.argmax(prediction)
        test_imgs, test_labels = next(test_generator)
        print("\nGround truth class: ", CLASSES[np.argmax(test_labels)])
        print("Predicted Class: ", CLASSES[cls_index])
    print("Model's accuracy on  test data: {:.0f}%".format(prediction_accuracy[1] * 100))

def main():
    predict()


if __name__ == '__main__':
    main()
