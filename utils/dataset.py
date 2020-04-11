import os
from torch.utils import data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class CatDogDataset(data.Dataset):

    def __init__(self, root_path, config, mode):

        self.config = config
        self.mode = mode

        if mode == 'test' or mode == 'valid':
            self.transforms = T.Compose([
                T.Resize((280, 280)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5])])

        elif mode == 'train':
            self.transforms = T.Compose([T.RandomRotation(30),
                                         T.RandomHorizontalFlip(),
                                         T.Resize((280, 280)),
                                         T.ToTensor(),
                                         T.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])])

        else:
            raise Exception('Error Mode.')

        self.dataset = ImageFolder(root=os.path.join(root_path), transform=self.transforms)

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = ImageFolder(root=os.path.join("../data/dogs-cats-images/dataset/training_set"))
    print(type(dataset))
    print(dataset[0:10])


