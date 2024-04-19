import tensorflow as tf

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv13123213'),
        tf.keras.layers.MaxPooling2D(2, 2, name='maxpool1123213'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2321321312'),
        tf.keras.layers.MaxPooling2D(2, 2, name='maxpoo21321312l2'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv32312312312'),
        tf.keras.layers.MaxPooling2D(2, 2, name='maxpoo12321312l3'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(512, activation='relu', name='d12312312ense1'),
        tf.keras.layers.Dropout(0.5, name='dropo12321321ut'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='o2312312utput')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
