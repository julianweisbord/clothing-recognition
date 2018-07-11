'''
Created on July 10th, 2018
Author: Julian Weisbord
Description: Create and train a CNN to classify articles of clothing.
'''

import tensorflow as tf

TRAIN_DATA = "./train"
VAL_DATA = "./val"
IMG_WIDTH = 300
IMG_HEIGHT = 300
BATCH_SIZE = 16
COLOR_CHANNELS = 3
KERNEL_SIZE = (3, 3)
N_EPOCHS = 40
SAMPLE_SIZE = 14000
VAL_SAMPLES = 3640
LEARNING_RATE = 0.001
CLASSES = ["Jeans", "Sweatpants", "Blazer"]


def load_data():
    '''
    Description: Create training dataset from train/val folders.
    Return: train_generator <Keras ImageDirectory Iterator>, validation_generator <Keras ImageDirectory Iterator>
    '''
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        TRAIN_DATA,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        classes=CLASSES,
        batch_size=BATCH_SIZE
    )
    validation_generator = datagen.flow_from_directory(
        VAL_DATA,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        classes=CLASSES,
        batch_size=BATCH_SIZE
    )
    return train_generator, validation_generator

def setup_model():
    '''
    Description: Create a sequential model and add layers.
    Return: model <Keras Sequential Model>
    '''
    model = tf.keras.models.Sequential()
    # Feature learning
    model.add(tf.keras.layers.Convolution2D(32, KERNEL_SIZE,
                                            activation='relu',
                                            input_shape=(IMG_WIDTH,
                                                         IMG_HEIGHT,
                                                         COLOR_CHANNELS)))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Convolution2D(32, KERNEL_SIZE, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Convolution2D(64, KERNEL_SIZE, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Classification
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.8))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    # Optimization
    model.compile(tf.keras.optimizers.Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train(model, train_generator, validation_generator):
    '''
    Description: Train the model with learning rate reduction.
    Input: model <Keras Sequential Model>, train_generator <Keras ImageDirectory Iterator>,
               validation_generator <Keras ImageDirectory Iterator>.
    '''
    lr_reduce_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                             patience=2,
                                                             verbose=1,
                                                             factor=.5,  # Reduce lr by half each time
                                                             min_lr=0.00001)

    model.fit_generator(
        train_generator,
        steps_per_epoch=SAMPLE_SIZE // BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=validation_generator,
        validation_steps=VAL_SAMPLES // BATCH_SIZE,
        callbacks=[lr_reduce_plateau])
    model_json = model.to_json()
    with open("models/model2.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('models/basic_cnn2.h5')
    print("saved model")

    print("Validation accuracy: ", model.evaluate_generator(validation_generator, VAL_SAMPLES)[1])

def main():
    train_generator, validation_generator = load_data()
    model = setup_model()
    train(model, train_generator, validation_generator)

if __name__ == '__main__':
    main()
