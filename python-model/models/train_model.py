import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from model import build_model
print(tf.__version__)


def calculate_steps(num_samples, batch_size):
    return (num_samples + batch_size - 1) // batch_size

def main():
    tf.keras.backend.clear_session()
    train_dir = './dataset/train'
    val_dir = './dataset/validation'
    target_size = (150, 150)
    batch_size = 16  # Consider adjusting if computational resources allow

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, class_mode='binary'
    )
    validation_generator = val_datagen.flow_from_directory(
        val_dir, target_size=target_size, batch_size=batch_size, class_mode='binary'
    )

    steps_per_epoch = calculate_steps(train_generator.samples, batch_size)
    validation_steps = calculate_steps(validation_generator.samples, batch_size)

    model = build_model((150, 150, 3))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    for epoch in range(200):  # Adjust based on when you see performance plateau
        print(f"Starting Epoch {epoch+1}")
        model.fit(
            x=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=1,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[early_stopping]
        )
        train_generator.on_epoch_end()
        validation_generator.on_epoch_end()

    # Save the model in Keras format to avoid compatibility issues
    model.save('hotdog_not_hotdog.h5')

    print(model.summary())

if __name__ == '__main__':
    main()
