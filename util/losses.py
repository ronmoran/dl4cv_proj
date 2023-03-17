from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F

from models.extractor import VitExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossG(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)

        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([global_resize_transform,
                                                    imagenet_norm
                                                    ])

        self.lambdas = dict(
            lambda_global_cls=cfg['lambda_global_cls'],
            lambda_global_ssim=0,
            lambda_entire_ssim=0,
            lambda_entire_cls=0,
            lambda_global_identity=0
        )

    def update_lambda_config(self, step):
        if step == self.cfg['cls_warmup']:
            self.lambdas['lambda_global_ssim'] = self.cfg['lambda_global_ssim']
            self.lambdas['lambda_global_identity'] = self.cfg['lambda_global_identity']

        if step % self.cfg['entire_A_every'] == 0:
            self.lambdas['lambda_entire_ssim'] = self.cfg['lambda_entire_ssim']
            self.lambdas['lambda_entire_cls'] = self.cfg['lambda_entire_cls']
        else:
            self.lambdas['lambda_entire_ssim'] = 0
            self.lambdas['lambda_entire_cls'] = 0

    def forward(self, outputs, inputs):
        self.update_lambda_config(inputs['step'])
        losses = {}
        loss_G = 0

        if self.lambdas['lambda_global_ssim'] > 0:
            losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs['x_global'], inputs['A_global'])
            loss_G += losses['loss_global_ssim'] * self.lambdas['lambda_global_ssim']

        if self.lambdas['lambda_entire_ssim'] > 0:
            losses['loss_entire_ssim'] = self.calculate_global_ssim_loss(outputs['x_entire'], inputs['A'])
            loss_G += losses['loss_entire_ssim'] * self.lambdas['lambda_entire_ssim']

        if self.lambdas['lambda_entire_cls'] > 0:
            losses['loss_entire_cls'] = self.calculate_crop_cls_loss(outputs['x_entire'], inputs['B_global'])
            loss_G += losses['loss_entire_cls'] * self.lambdas['lambda_entire_cls']

        if self.lambdas['lambda_global_cls'] > 0:
            losses['loss_global_cls'] = self.calculate_crop_cls_loss(outputs['x_global'], inputs['B_global'])
            loss_G += losses['loss_global_cls'] * self.lambdas['lambda_global_cls']

        if self.lambdas['lambda_global_identity'] > 0:
            losses['loss_global_id_B'] = self.calculate_global_id_loss(outputs['y_global'], inputs['B_global'])
            loss_G += losses['loss_global_id_B'] * self.lambdas['lambda_global_identity']

        losses['loss'] = loss_G
        return losses

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            keys_ssim = self.calculate_global_ssim(a, False)
            target_keys_self_sim = self.calculate_global_ssim(b, True)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_global_ssim(self, input_im, skip_grad):
        input_im = self.global_transform(input_im)
        if skip_grad:
            with torch.no_grad():
                return self.extractor.get_keys_self_sim_from_input(input_im.unsqueeze(0), layer_num=11)
        else:
            return self.extractor.get_keys_self_sim_from_input(input_im.unsqueeze(0), layer_num=11)

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            cls_token = self.calculate_cls_token(a, False)
            target_cls_token = self.calculate_cls_token(b, True)

            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_cls_token(self, im, skip_grad, to_device=True):
        im = self.global_transform(im).unsqueeze(0)
        if to_device:
            im = im.to(device)
        if skip_grad:
            with torch.no_grad():
                return self.extractor.get_feature_from_input(im)[-1][0, 0, :]
        else:
            return self.extractor.get_feature_from_input(im)[-1][0, 0, :]


    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
            keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss
