import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

def split_dataset(dataset : list) -> np.array, np.array, np.array, np.array:
    x : list = list([])
    y : list = list([])

    for data in dataset:
        x.append(data['data'])
        y.append(data['label'])
    
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = .2, shuffle = True, random_state = 42)

    return train_x, train_y, test_x, test_y

class AnimalClassifier(BaseEstimator):
    def __init__(self):
        pass