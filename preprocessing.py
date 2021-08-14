import os
import cv2
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
            result[animal].append(cv2.imread(f))

    return result

def preprocess_image(img : np.ndarray) -> np.ndarray:
    # Gray scale:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize:
    img = cv2.resize(img, (256, 256))
    # Gaussian blur:
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def compute_hog(img : np.ndarray, number_blocks : int = 4, number_bins : int = 12) -> np.ndarray:
    winSize : tuple = img.shape
    blockSize : tuple = (int(winSize[0] / number_blocks), int(winSize[1] / number_blocks))
    blockStride : tuple = blockSize
    cellSize : tuple = (int(blockSize[0] / 2), int(blockSize[1] / 2))
    hog : cv2.HOGDescriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, number_bins)
    vector : np.ndarray = hog.compute(img)
    return np.reshape(vector, (vector.shape[0]))

def compute_lbp(img : np.ndarray, radius : int = 1, number_points : int = 8, METHOD : str = 'uniform') -> np.ndarray:
    lbp = local_binary_pattern(img, number_points, radius, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins = np.arrange(0, 255))
    hist = hist.astype('float')
    hist /= hist.sum()
    return hist

def compute_sift(img : np.ndarray, number_features : int = 0, octave_layers : int = 3, contrast_threshold : float = .04, edge_threshold : float = 10, sigma : float = 1.6, descriptor_type : int = cv2.CV_32F) -> np.ndarray:
    sift : xf2d.SIFT = cv2.xfeatures2d.SIFT_create(number_features, octave_layers, contrast_threshold, edge_threshold, sigma, descriptor_type)

    keypoints, descriptors = sift.detectAndCompute(img, None, useProvidedDescriptors = True)
    
    return descriptors.astype('float')

def feature_extraction(img_dict : dict) -> list:
    results : list = list([])
    
    for animal, img_list in img_dict.items():
        result = dict({})
        for img in img_list:
            result['hog'] = compute_hog(img)
            result['lbp'] = compute_lbp(img)
            result['sift'] = compute_sift(img)
            result['label'] = animal
        results.append(result)

    return results
