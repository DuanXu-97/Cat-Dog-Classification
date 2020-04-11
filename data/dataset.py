import os
from torch.utils import data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class CatDogDataset(data.Dataset):

    def __init__(self, root_path, config, mode):

        self.config = config
        self.mode = mode

        self.dataset = ImageFolder(root=os.path.join(root_path))

        if mode == 'test':
            self.image_nums = config.test_image_nums

        elif mode == 'train' or mode == 'valid':
            self.image_nums = config.train_image_nums

        else:
            raise Exception('Error Mode.')

        if mode == 'train':
            self.image_nums = config.train_image_nums * 0.7
            self.dataset = self.dataset[:int(config.train_image_nums * 0.7)]
        elif mode == 'valid':
            self.image_nums = config.train_image_nums - config.train_image_nums * 0.7
            self.dataset = self.dataset[int(config.train_image_nums * 0.7):]

    def __getitem__(self, index):
        return T.ToTensor()(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


