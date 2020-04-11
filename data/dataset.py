import os
from torch.utils import data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class CatDogDataset(data.Dataset):

    def __init__(self, root_path, config, mode):

        self.config = config
        self.mode = mode

        if mode == 'test':
            self.image_nums = config.test_image_nums
            dataset = ImageFolder(root=os.path.join(root_path, 'test_set'))
            self.data = dataset[:][0]
            self.labels = dataset[:][1]

        elif mode == 'train' or mode == 'valid':
            self.image_nums = config.train_image_nums
            dataset = ImageFolder(root=os.path.join(root_path, 'training_set'))
            self.data = dataset[:][0]
            self.labels = dataset[:][1]

            if mode == 'train':
                self.image_nums = config.train_image_nums * 0.7
                self.data = self.data[:int(config.train_image_nums * 0.7)]
                self.labels = self.labels[:int(config.train_image_nums * 0.7)]
            elif mode == 'valid':
                self.image_nums = config.train_image_nums - config.train_image_nums * 0.7
                self.data = self.data[int(config.train_image_nums * 0.7):]
                self.labels = self.labels[int(config.train_image_nums * 0.7):]

        else:
            raise Exception('Error Mode.')

    def __getitem__(self, index):
        return T.ToTensor()(self.data[index]), self.labels[index]

    def __len__(self):
        return self.data.shape[0]


