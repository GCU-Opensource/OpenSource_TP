import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Data:

    def __init__(self, root, face, mask):
        self.root = root
        self.face_path = os.path.join(root, face)
        self.mask_path = os.path.join(root, mask)

    def RMFD_dataset_handler(self):

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        face_dataset = RMFD_Face(self.root, self.face_path, transform=transform)
        mask_dataset = RMFD_Mask(self.root, self.mask_path, transform=transform)

        # divide dataset into train and test
        from torch.utils.data import random_split
        train_size = int(0.8 * len(face_dataset))
        test_size = len(face_dataset) - train_size
        train_face, test_face = random_split(face_dataset, [train_size, test_size])

        train_size = int(0.8 * len(mask_dataset))
        test_size = len(mask_dataset) - train_size
        train_mask, test_mask = random_split(mask_dataset, [train_size, test_size])

        # merge train_face and train_mask
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([train_face, train_mask])
        test_dataset = ConcatDataset([test_face, test_mask])

        # create dataloader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        return train_loader, test_loader

# Path: datasets/data_loader.py


class RMFD_Face(Dataset):
    def __init__(self, root, face, transform=None):
        self.root = root
        self.face_path = face
        self.transform = transform
        self.face_list = os.listdir(self.face_path)

        # each face_list and mask_list is folder
        # therefore, we should iterate through each folder
        # and get the list of images
        self.face_list = [os.path.join(self.face_list[i], j) for i in range(len(self.face_list)) for j in
                          os.listdir(os.path.join(self.face_path, self.face_list[i]))]

        self.face_list.sort()

    def __len__(self):
        return len(self.face_list)

    def __getitem__(self, idx):
        # label 0 for face
        label = 0

        # get image
        img_name = os.path.join(self.face_path, self.face_list[idx])
        image = Image.open(img_name).convert('RGB')

        # transform
        if self.transform:
            image = self.transform(image)

        return image, label


class RMFD_Mask(Dataset):
    def __init__(self, root, mask, transform=None):
        self.root = root
        self.mask_path = mask
        self.transform = transform
        self.mask_list = os.listdir(self.mask_path)

        # each face_list and mask_list is folder
        # therefore, we should iterate through each folder
        # and get the list of images
        self.mask_list = [os.path.join(self.mask_list[i], j) for i in range(len(self.mask_list)) for j in os.listdir(os.path.join(self.mask_path, self.mask_list[i]))]

        self.mask_list.sort()

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        # label 1 for mask
        label = 1

        # get image
        img_name = os.path.join(self.mask_path, self.mask_list[idx])
        image = Image.open(img_name).convert('RGB')

        # transform
        if self.transform:
            image = self.transform(image)

        return image, label