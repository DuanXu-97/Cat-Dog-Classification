import os
from torch.utils import data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class CatDogDataset(data.Dataset):

    def __init__(self, root_path, config, mode):

        self.config = config
        self.mode = mode
        self.dataset = list()

        _dataset = ImageFolder(root=os.path.join(root_path))

        if mode == 'test':
            self.image_nums = config.test_image_nums

        elif mode == 'train' or mode == 'valid':
            self.image_nums = config.train_image_nums

        else:
            raise Exception('Error Mode.')

        if mode == 'train':
            self.image_nums = int(config.train_image_nums * 0.8)
            for i in range(0, self.image_nums):
                self.dataset.append(_dataset[i])
        elif mode == 'valid':
            self.image_nums = int(config.train_image_nums - config.train_image_nums * 0.8)
            for i in range(int(config.train_image_nums * 0.8), config.train_image_nums):
                self.dataset.append(_dataset[i])
    def __getitem__(self, index):
        return T.ToTensor()(T.Resize([400, 400])(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = ImageFolder(root=os.path.join("../data/dogs-cats-images/dataset/training_set"))
    print(type(dataset))
    print(dataset[0:10])


