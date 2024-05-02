import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_prepare_image(filepath, target_size=(150, 150)):
    img = image.load_img(filepath, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Model expects 4D tensor
    return img_array / 255.

def predict(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img_array = load_and_prepare_image(image_path)
    prediction = model.predict(img_array)
    return prediction[0][0]

if __name__ == '__main__':
    model_path = 'hotdog_not_hotdog.h5'
    image_path = 'dataset/h2.jpg'
    prediction = predict(model_path, image_path)
    print(f"Prediction: {prediction}")
    if prediction > 0.5:
        print("It's a hotdog!")
    else:
        print("It's not a hotdog.")
