from pathlib import Path
import tensorflow as tf
model = tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3))
export_path = Path.cwd() / "saved_models" / "MobileNetV3Large"
model.save(export_path)