import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os, re
from glob import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as torch_multiprocessing


def data_transform(img_size):
    transform_list = [
        transforms.Resize(size=[img_size, img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # [0, 1] -> [-1, 1]
    ]
    return transforms.Compose(transform_list)


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


def check_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def find_latest_ckpt(folder):
    """ find latest checkpoint """
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        file = max(files)[1]
        file_name = os.path.splitext(file)[0]
        previous_iter = int(file_name.split("_")[1])
        return file, previous_iter
    else:
        return None, 0


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def apply_gradients(loss, optim):
    optim.zero_grad()
    loss.backward()
    optim.step()


def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cross_entroy_loss(logit, label):
    loss = torch.nn.CrossEntropyLoss()(logit, label)
    return loss


def reduce_loss(tmp):
    """ will implement reduce_loss func """
    loss = tmp
    return loss
# def reduce_loss_dict(loss_dict):
#     world_size = get_world_size()
#
#     if world_size < 2:
#         return loss_dict
#
#     with torch.no_grad():
#         keys = []
#         losses = []
#
#         for k in sorted(loss_dict.keys()):
#             keys.append(k)
#             losses.append(loss_dict[k])
#
#         losses = torch.stack(losses, 0)
#         dist.reduce(losses, dst=0)
#
#         if dist.get_rank() == 0:
#             losses /= world_size
#
#         reduced_losses = {k: v.mean().item() for k, v in zip(keys, losses)}
#
#     return reduced_losses
