import os.path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import os.path
import torch
import shutil

from data.transforms import Global_crops, dino_structure_transforms, dino_texture_transforms


class SingleImageDataset(Dataset):
    def __init__(self, cfg, b_name):
        self.cfg = cfg
        self.structure_transforms = dino_structure_transforms if cfg['use_augmentations'] else transforms.Compose([])
        self.texture_transforms = dino_texture_transforms if cfg['use_augmentations'] else transforms.Compose([])
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.global_A_patches = transforms.Compose(
            [
                self.structure_transforms,
                Global_crops(n_crops=cfg['global_A_crops_n_crops'],
                             min_cover=cfg['global_A_crops_min_cover'],
                             last_transform=self.base_transform)
            ]
        )

        self.global_B_patches = transforms.Compose(
            [
                self.texture_transforms,
                Global_crops(n_crops=cfg['global_B_crops_n_crops'],
                             min_cover=cfg['global_B_crops_min_cover'],
                             last_transform=self.base_transform)
            ]
        )

        # open images
        self.A_img = self.__read_img("A")
        self.B_img = self.__read_img("B", b_name)
        self.__transform_imgs()

        print("Image sizes %s and %s" % (str(self.A_img.size), str(self.B_img.size)))
        self.step = torch.zeros(1) - 1

    def get_A(self):
        return self.base_transform(self.A_img).unsqueeze(0)

    def __transform_imgs(self):
        if self.cfg['A_resize'] > 0:
            self.A_img = transforms.Resize(self.cfg['A_resize'])(self.A_img)

        if self.cfg['B_resize'] > 0:
            self.B_img = transforms.Resize(self.cfg['B_resize'])(self.B_img)

        if self.cfg['direction'] == 'BtoA':
            self.A_img, self.B_img = self.B_img, self.A_img

    def __read_img(self, a_or_b, b_name=None):
        return Image.open(self.__get_img_path(a_or_b, b_name)).convert('RGB')

    def __get_img_path(self, a_or_b, b_name=None):
        img_dir = os.path.join(self.cfg['dataroot'], a_or_b)
        if b_name:
            img_name = os.path.join(img_dir, b_name)
        else:
            img_name = os.listdir(img_dir)[0]
        return os.path.join(img_dir, img_name)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        self.step += 1
        sample = {'step': self.step}
        if self.step % self.cfg['entire_A_every'] == 0:
            sample['A'] = self.get_A()
        sample['A_global'] = self.global_A_patches(self.A_img)
        sample['B_global'] = self.global_B_patches(self.B_img)

        return sample


class StructureImageDataSet(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.gs_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Grayscale(3)])
        self.A_img = self.__read_img()

    def __read_img(self):
        return Image.open(self.__get_img_path("A")).convert('RGB')

    def __get_img_path(self, a_or_else):
        if a_or_else == "A":
            img_dir = os.path.join(self.cfg['dataroot'], a_or_else)
            img_name = os.listdir(img_dir)[0]
        else:
            img_dir = os.path.join(self.cfg['dataroot'], "B")
            img_name = os.path.basename(a_or_else)
        return os.path.join(img_dir, img_name)

    def replace_appearance_img(self, new_img_path):
        file_name = self.__get_img_path(new_img_path)
        shutil.copy(new_img_path, file_name)
        return file_name

    def __len__(self):
        return 1

    def __getitem__(self, item) -> torch.Tensor:
        if item != 0:
            raise ValueError("Dataset has only one fixed image")
        return self.gs_transform(self.A_img)
