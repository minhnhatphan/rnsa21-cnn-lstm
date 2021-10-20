import torch
import os
import random
import numpy as np
import cv2
import pydicom

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_image(path, size=(256,256)):
    image = cv2.imread(path, 0)
    if image is None:
        return np.zeros()
    
    image = cv2.resize(image, size) / 255
    return image.astype('f')

def uniform_temporal_subsample(x, num_samples):
    '''
        Modified from https://github.com/facebookresearch/pytorchvideo/blob/d7874f788bc00a7badfb4310a912f6e531ffd6d3/pytorchvideo/transforms/functional.py#L19
        Args:
            x: input list
            num_samples: The number of equispaced samples to be selected
        Returns:
            Output list     
    '''
    t = len(x)
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return [x[i] for i in indices]

def load_dicom(path, img_size):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
        
    data = np.float32(cv2.resize(data, (img_size, img_size)))
    return torch.tensor(data)

class LossMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        # incremental update
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg


class AccMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0
        
    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().astype(int)
        y_pred = y_pred.cpu().numpy() >= 0
        last_n = self.n
        self.n += len(y_true)
        true_count = np.sum(y_true == y_pred)
        # incremental update
        self.avg = true_count / self.n + last_n / self.n * self.avg