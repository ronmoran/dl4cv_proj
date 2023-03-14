import torch
import numpy as np
import random
from data.Dataset import SingleImageDataset, StructureImageDataSet
from models.model import Model
from util.losses import LossG
from util.util import get_scheduler, get_optimizer, save_result
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from json import load
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_similar_appearance_img(dataset: StructureImageDataSet, criterion: LossG,
                                style_tokens_path: str, train_imgs_path: str):
    appearance = dataset[0]
    criterion.extractor.model = criterion.extractor.model.to("cpu")
    appearance_cls_token = criterion.calculate_cls_token(appearance, True, False)
    criterion.extractor.model = criterion.extractor.model.to(device)
    with open(os.path.join(style_tokens_path, "paths.json")) as f:
        paths = load(f)
    tokens = torch.load(os.path.join(style_tokens_path, "tokens.pt"), map_location="cpu")
    cosine_similarity_sorted = torch.cosine_similarity(tokens, appearance_cls_token).sort()
    img_index = cosine_similarity_sorted.indices[-1]  # Last index is most similar
    new_appearance_path = paths[img_index]  # most similar image
    print(f"New appearance image is: {new_appearance_path}")
    dataset.replace_appearance_img(os.path.join(train_imgs_path, new_appearance_path))


def train_model(dataroot, style: str, tokens_path, train_imgs_path, callback=None, output_file_prefix=''):
    with open("conf/default/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    cfg = config

    if dataroot is not None:
        cfg['dataroot'] = dataroot

    # set seed
    seed = cfg['seed']
    if seed == -1:
        seed = np.random.randint(2 ** 32 - 1, dtype=np.int64)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    print(f'running with seed: {seed}.')

    # define loss function
    criterion = LossG(cfg)

    # copy appearance img
    find_similar_appearance_img(StructureImageDataSet(cfg), criterion,
                                os.path.join(tokens_path, style), train_imgs_path)

    # create dataset, loader
    dataset = SingleImageDataset(cfg)

    # define model
    model = Model(cfg)

    # define optimizer, scheduler
    optimizer = get_optimizer(cfg, model.netG.parameters())

    scheduler = get_scheduler(optimizer,
                              lr_policy=cfg['scheduler_policy'],
                              n_epochs=cfg['n_epochs'],
                              n_epochs_decay=cfg['scheduler_n_epochs_decay'],
                              lr_decay_iters=cfg['scheduler_lr_decay_iters'])

    with tqdm(range(1, cfg['n_epochs'] + 1)) as tepoch:
        for epoch in tepoch:
            inputs = dataset[0]
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion(outputs, inputs)
            loss_G = losses['loss']
            log_data = losses
            log_data['epoch'] = epoch

            # update learning rate
            lr = optimizer.param_groups[0]['lr']
            log_data["lr"] = lr
            tepoch.set_description(f"Epoch {log_data['epoch']}")
            tepoch.set_postfix(loss=log_data["loss"].item(), lr=log_data["lr"])

            # log current generated entire image
            if epoch % cfg['log_images_freq'] == 0:
                img_A = dataset.get_A().to(device)
                with torch.no_grad():
                    output = model.netG(img_A)
                # save_result(output[0], cfg['dataroot'])
                if callback is not None:
                    callback(output[0])
            # every 1000 epochs save the output separately
            if epoch % 1000 == 0:
                img_A = dataset.get_A().to(device)
                with torch.no_grad():
                    output = model.netG(img_A)
                save_result(output[0], cfg['dataroot'], f'output_{output_file_prefix}_{epoch}')

            loss_G.backward()
            optimizer.step()
            scheduler.step()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataroot", type=str)
    args = parser.parse_args()
    dataroot = args.dataroot

    train_model(dataroot)
