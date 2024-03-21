import os
from glob import glob

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, img_size, dataset_path):
        self.train_images = self.listdir(dataset_path)
        self.train_labels = list(pd.read_csv(dataset_path + '/annotations.csv', header=None).iloc[:, 1])

        # interpolation=transforms.InterpolationMode.BICUBIC, antialias=True

        self.transform = data_transform(img_size)

    def listdir(self, dir_path):
        extensions = ['png', 'jpg']
        file_path = []
        for ext in extensions:
            file_path += glob(os.path.join(dir_path, '*.' + ext))
        file_path.sort()
        return file_path

    def __getitem__(self, index):
        sample_path = self.train_images[index]
        img = Image.open(sample_path).convert('RGB')
        img = self.transform(img)

        # 여기서 문제
        label = self.train_labels[index]

        return img, label

    def __len__(self):
        return len(self.train_images)


def data_transform(img_size):
    transform_list = [
        transforms.Resize(size=[img_size, img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # [0, 1] -> [-1, 1]
    ]
    return transforms.Compose(transform_list)
