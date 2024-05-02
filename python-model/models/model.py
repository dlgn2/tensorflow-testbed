import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Dropout

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same', name='conv1'),
        tf.keras.layers.MaxPooling2D(2, 2, name='max_pool1'),
        BatchNormalization(name='batch_norm1'),
        tf.keras.layers.Dropout(0.25, name='dropout1'),  # New dropout layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        tf.keras.layers.MaxPooling2D(2, 2, name='max_pool2'),
        BatchNormalization(name='batch_norm2'),
        tf.keras.layers.Dropout(0.25, name='dropout2'),  # New dropout layer
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        tf.keras.layers.MaxPooling2D(2, 2, name='max_pool3'),
        BatchNormalization(name='batch_norm3'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4'),  # New Conv Layer
        tf.keras.layers.MaxPooling2D(2, 2, name='max_pool4'),
        BatchNormalization(name='batch_norm4'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', name='dense1'),
        tf.keras.layers.Dropout(0.5, name='dropout_final'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
