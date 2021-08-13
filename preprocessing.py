import os
import cv2 as cv
import numpy as np
from glob import glob

def import_images(animal_dir_names : dict) -> dict:
    result : dict = dict({})
    
    for animal, dir_name in animal_dir_names.items():
        result[animal] = list(os.listdir('raw-img/' + dir_name))

    return result

def preprocess_image(img : np.ndarray) -> np.ndarray:
    # Gray scale:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Resize:
    img = cv.resize(img, (256, 256))
    # Gaussian blur:
    img = cv.GaussianBlur(img, (5, 5), 0)
    return img

def feature_extraction() -> (np.ndarray, np.ndarray):
    # HOG
    # Haar
    # SIFT
    # SURF
    pass
