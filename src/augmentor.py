import tensorflow as tf
from tensorflow.keras import layers

def get_augmentation_model():
    model = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),     
    ])
    return model