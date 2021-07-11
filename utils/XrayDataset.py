import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import os
class ChestXRayDataSet:
    def __init__(self, path:str, transforms = None):
        self.img_path = path
        self.transforms = transforms
    def __call__(self):
        return datasets.ImageFolder(self.img_path, self.transforms)

# path_ = ".\chest_xray_data"
# mode = "train"
# train_path = os.path.join(path_,mode)
# print(train_path)
# train_set = ChestXRayDataSet(train_path, None)
# train_data = train_set()
# print(len(train_data))