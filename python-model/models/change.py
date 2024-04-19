import tensorflowjs as tfjs

# Load your existing Keras model
from tensorflow.keras.models import load_model
model = load_model('hotdog_not_hotdog.keras')

# Convert and save as TensorFlow.js format
tfjs.converters.save_keras_model(model, 'output_directory/')
