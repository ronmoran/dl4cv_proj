import os.path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import os.path
import torch

from data.transforms import Global_crops, dino_structure_transforms, dino_texture_transforms


class SingleImageDataset(Dataset):
    def __init__(self, cfg):
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
        dir_A = os.path.join(cfg['dataroot'], 'A')
        dir_B = os.path.join(cfg['dataroot'], 'B')
        A_path = os.listdir(dir_A)[0]
        B_path = os.listdir(dir_B)[0]
        self.A_img = Image.open(os.path.join(dir_A, A_path)).convert('RGB')
        self.B_img = Image.open(os.path.join(dir_B, B_path)).convert('RGB')

        if cfg['A_resize'] > 0:
            self.A_img = transforms.Resize(cfg['A_resize'])(self.A_img)

        if cfg['B_resize'] > 0:
            self.B_img = transforms.Resize(cfg['B_resize'])(self.B_img)

        if cfg['direction'] == 'BtoA':
            self.A_img, self.B_img = self.B_img, self.A_img

        print("Image sizes %s and %s" % (str(self.A_img.size), str(self.B_img.size)))
        self.step = torch.zeros(1) - 1

    def get_A(self):
        return self.base_transform(self.A_img).unsqueeze(0)

    def __getitem__(self, index):
        self.step += 1
        sample = {'step': self.step}
        if self.step % self.cfg['entire_A_every'] == 0:
            sample['A'] = self.get_A()
        sample['A_global'] = self.global_A_patches(self.A_img)
        sample['B_global'] = self.global_B_patches(self.B_img)

        return sample

    def __len__(self):
        return 1


class WikiArtClassifyDataset(ImageFolder):

    def __init__(self, cfg):
        global_resize_transform = transforms.Resize(cfg['dino_global_patch_size'], max_size=480)
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        texture_transforms = dino_texture_transforms if cfg['use_augmentations'] else transforms.Compose([])
        cls_transforms = transforms.Compose([texture_transforms, transforms.ToTensor(), global_resize_transform, imagenet_norm])
        super().__init__(cfg["dataroot"], cls_transforms)
