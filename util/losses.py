from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F
import os

from models.extractor import VitExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossG(torch.nn.Module):
    STYLES = {style_name: i for i, style_name in enumerate(['Cubism', 'Impressionism', 'Naïve Art (Primitivism)', 'Rococo', 'Ukiyo-e'])}

    def __init__(self, cfg, target_class=None):
        super().__init__()

        self.cfg = cfg
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)
        classifier_name = 'dino_class.pt'  # 'resnet18_ft.pt', 'dino_class.pt''dino_classifier.pt'
        print(f"Using classifier: {classifier_name}")
        self.classifier = torch.load(os.path.join(os.getcwd(), 'models', classifier_name), map_location=device)
        self.classifier.eval()  # TODO: is this enough to freeze the weights?
        assert target_class in self.STYLES
        # self.target_classification = torch.eye(len(self.STYLES))[self.STYLES[target_class]]  # for F.cross_entropy
        self.target_classification = torch.tensor(self.STYLES[target_class]).unsqueeze(0).to(device)

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
            lambda_global_identity=0,
            lambda_entire_classifier=0,
            lambda_global_classifier=0
        )

    def update_lambda_config(self, step):
        if step == self.cfg['cls_warmup']:
            self.lambdas['lambda_global_ssim'] = self.cfg['lambda_global_ssim']
            self.lambdas['lambda_global_identity'] = self.cfg['lambda_global_identity']
            self.lambdas['lambda_global_classifier'] = self.cnf['lambda_global_classifier']

        if step % self.cfg['entire_A_every'] == 0:
            self.lambdas['lambda_entire_ssim'] = self.cfg['lambda_entire_ssim']
            self.lambdas['lambda_entire_cls'] = self.cfg['lambda_entire_cls']
            self.lambdas['lambda_entire_classifier'] = self.conf['lambda_entire_classifier']
        else:
            self.lambdas['lambda_entire_ssim'] = 0
            self.lambdas['lambda_entire_cls'] = 0
            self.lambdas['lambda_entire_classifier'] = 0

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

        if self.lambdas['lambda_entire_classifier'] > 0:
            losses['loss_entire_cls'] = self.calculate_crop_classification_loss(outputs['x_entire'])
            loss_G += losses['loss_entire_cls'] * self.lambdas['lambda_entire_cls']

        if self.lambdas['lambda_global_classifier'] > 0:
            losses['loss_global_cls'] = self.calculate_crop_classification_loss(outputs['x_global'])
            loss_G += losses['loss_global_cls'] * self.lambdas['lambda_global_cls']

        losses['loss'] = loss_G
        return losses

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(device)
            b = self.global_transform(b).unsqueeze(0).to(device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_crop_classification_loss(self, outputs):
        loss = 0.0
        for a in outputs:  # use same transformation as original loss functions
            a = self.global_transform(a).unsqueeze(0).to(device)
            a_classification = self.classifier(a)
            loss += F.cross_entropy(a_classification, self.target_classification)
        return loss

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
