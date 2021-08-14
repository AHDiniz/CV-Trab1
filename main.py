import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from preprocessing import import_images, feature_extraction

animal_dir_names = dict({
    'dog': 'cane',
    'horse': 'cavallo',
    'elephant': 'elefante',
    'butterfly': 'farfalla',
    'chicken': 'gallina',
    'cat': 'gatto',
    'cow': 'mucca',
    'sheep': 'pecora',
    'spider': 'ragno',
    'squirrel': 'scoiattolo'
})

# Preprocessing and feature extraction:
features = feature_extraction(import_images(animal_dir_names))

# Train the classification model
# Get training statistical data
# Test the classification model
# Get testing statistical model
