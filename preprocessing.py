import os
import cv2 as cv
import numpy as np
from glob import glob
from skimage.feature import local_binary_pattern

def import_images(animal_dir_names : dict) -> dict:
    result : dict = dict({})
    
    for animal, dir_name in animal_dir_names.items():
        file_list : list = list(os.listdir('raw-img/' + dir_name))
        result[animal] = list([])
        for filename in file_list:
            f = os.path.join('raw-img/' + dir_name, filename)
            result[animal].append(cv.imread(f))

    return result

def preprocess_image(img : np.ndarray) -> np.ndarray:
    # Gray scale:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Resize:
    img = cv.resize(img, (256, 256))
    # Gaussian blur:
    img = cv.GaussianBlur(img, (5, 5), 0)
    return img

def compute_hog(img : np.ndarray, number_blocks : int = 4, number_bins : int = 12) -> np.ndarray:
    winSize : tuple = img.shape
    blockSize : tuple = (int(winSize[0] / number_blocks), int(winSize[1] / number_blocks))
    blockStride : tuple = blockSize
    cellSize : tuple = (int(blockSize[0] / 2), int(blockSize[1] / 2))
    hog : cv.HOGDescriptor = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, number_bins)
    vector : np.ndarray = hog.compute(img)
    return np.reshape(vector, (vector.shape[0]))

def compute_lbp(img : np.ndarray, radius : int = 1, number_points : int = 8, METHOD : str = 'uniform') -> np.ndarray:
    lbp = local_binary_pattern(img, number_points, radius, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins = np.arrange(0, 255))
    hist = hist.astype('float')
    hist /= hist.sum()
    return hist

def feature_extraction(img_dict : dict) -> (np.ndarray, np.ndarray):
    for animal, img_list in img_dict.items():
        for img in img_list:
            hog_vector = compute_hog(img)
            lbp_hist = compute_lbp(img)
    pass
