import torch

from models.extractor import VitExtractor
from models.networks import init_net

from torchvision import transforms
from torch import nn
from typing import Union


class ViTClassifier(nn.Module):
    def __init__(self, model_name: str, device: Union[str, torch.device], hidden_layer_size: int, num_classes: int,
                 dino_global_patch_size: int, init_type: str, init_gain: Union[float, None] = None):
        super().__init__()
        self.extractor = VitExtractor(model_name, device)
        self.classifier_head = nn.Sequential(nn.Linear(self.extractor.get_embedding_dim(), hidden_layer_size),
                                             nn.Dropout(0.2),
                                             nn.ReLU(),
                                             nn.Linear(hidden_layer_size, num_classes)).to(device)
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = transforms.Resize(dino_global_patch_size, max_size=480)

        self.global_transform = transforms.Compose([global_resize_transform, imagenet_norm])
        self.__init_type = init_type
        self.__init_gain = init_gain
        self.init_weights()

    def init_weights(self):
        self.classifier_head = init_net(self.classifier_head, self.__init_type, self.__init_gain)

    def forward(self, inp: torch.Tensor, device: Union[None, torch.device] = None):
        """
        :param inp: Of size B x C x H x W where B is batch size, C is channels (should be 3), H is image height
        and w is image width. MUST be on the same device as this module
        :param device: If given, moves this module and the given tensor to device. Otherwise, does not move at all.
        """
        inp = self.global_transform(inp)
        if device is not None:
            self.to(device)
            self.extractor.model.to(device)
            inp = inp.to(device)
        if self.training:  # During training, don't calculate gradients for dino
            with torch.no_grad():
                cls_token = self.__extract_cls_token(inp)
        else:
            cls_token = self.__extract_cls_token(inp)  # When running splice, gradients need to be computed

        return self.classifier_head(cls_token)

    def train(self, mode: bool = True):
        super().train(mode)
        self.extractor.model.eval()
        return self

    def __extract_cls_token(self, inp):
        return self.extractor.get_feature_from_input(inp)[-1][:, 0, :]
