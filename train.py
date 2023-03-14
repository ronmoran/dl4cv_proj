import torch
import torch.utils.data
import numpy as np
import random
import copy
import time
from data.Dataset import SingleImageDataset, WikiArtClassifyDataset
from models import Model, ViTClassifier
from util.losses import LossG
from util.util import get_scheduler, get_optimizer, save_result
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def config_boilerplate(dataroot: str) -> dict:

    with open("conf/default/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if dataroot is not None:
        config['dataroot'] = dataroot

    # set seed
    seed = config['seed']
    if seed == -1:
        seed = np.random.randint(2 ** 32 - 1, dtype=np.int64)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    print(f'running with seed: {seed}.')
    return config


def train_classifier(dataroot: str, save_model_path: Union[str, None]):
    # read config yaml
    cfg = config_boilerplate(dataroot)

    # create dataset, loader
    dataset = WikiArtClassifyDataset(cfg)

    # define model
    model = ViTClassifier(cfg["dino_model_name"], device, cfg["dino_classifier_num_hidden"],
                          cfg["dino_style_cls_count"], cfg["dino_global_patch_size"],
                          cfg["init_type"], cfg["init_gain"])

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # define optimizer, scheduler
    optimizer = get_optimizer(cfg, model.classifier_head.parameters())

    scheduler = get_scheduler(optimizer,
                              lr_policy=cfg['scheduler_policy'],
                              n_epochs=cfg['n_epochs'],
                              n_epochs_decay=cfg['scheduler_n_epochs_decay'],
                              lr_decay_iters=cfg['scheduler_lr_decay_iters'])

    train_size = int(len(dataset) * 0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, (train_size, len(dataset) - train_size))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}

    # Initializing
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Train and evaluate
    for _ in tqdm(range(1, cfg['n_epochs'] + 1)):
        for phase in ['train', 'val']:

            # For statistics
            running_loss = 0.0
            running_corrects = 0

            # Set model phase
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data and run in model
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                print(f"{i}/{len(dataloaders[phase])}\r", end="")
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()  # reset gradients
                with torch.set_grad_enabled(model.training):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                _, preds = torch.max(outputs, 1)  # what?
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if epoch_acc > best_acc and phase == "val":
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # Finalize and print training info
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    if save_model_path is not None:
        torch.save(model, save_model_path)
    return model


def train_model(dataroot, callback=None, style=None, output_file_prefix=''):

    if style is not None:
        assert style in LossG.STYLES, f'style should be one of {LossG.STYLES.keys()}.'
    style = 'Ukiyo-e' if style is None else style
    print(f'Aiming for {style} style.\n')

    # read config yaml
    cfg = config_boilerplate(dataroot)
    # create dataset, loader
    dataset = SingleImageDataset(cfg)

    # define model
    model = Model(cfg)

    # define loss function
    criterion = LossG(cfg, target_class=style)

    # define optimizer, scheduler
    optimizer = get_optimizer(cfg, model.netG.parameters())

    scheduler = get_scheduler(optimizer,
                              lr_policy=cfg['scheduler_policy'],
                              n_epochs=cfg['n_epochs'],
                              n_epochs_decay=cfg['scheduler_n_epochs_decay'],
                              lr_decay_iters=cfg['scheduler_lr_decay_iters'])

    global_classification_losses = []
    entire_classification_losses = []
    global_ssim_losses = []
    entire_ssim_losses = []

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

            # if criterion.target_classification == criterion.curr_classification:
            #     print(f'Classified correctly in epoch {epoch}')

            # log current generated entire image
            if epoch % cfg['log_images_freq'] == 0:
                img_A = dataset.get_A().to(device)
                with torch.no_grad():
                    output = model.netG(img_A)
                # save_result(output[0], cfg['dataroot'], f'output_{output_file_prefix}')
                if callback is not None:
                    callback(output[0])
            # every 1000 epochs save the output separately
            if epoch % 1000 == 0:
                img_A = dataset.get_A().to(device)
                with torch.no_grad():
                    output = model.netG(img_A)
                save_result(output[0], cfg['dataroot'], f'output_{output_file_prefix}_{epoch}')

            global_classification_losses.append(losses['loss_global_cls'])
            entire_classification_losses.append(losses['loss_entire_cls'])
            global_ssim_losses.append(losses['loss_global_ssim'])
            entire_ssim_losses.append(losses['loss_entire_ssim'])


            loss_G.backward()
            optimizer.step()
            scheduler.step()

    return {'global_class_loss': global_classification_losses,
            'entire_class_loss': entire_classification_losses,
            'global_ssim_loss': global_ssim_losses,
            'entire_ssim_loss': entire_ssim_losses}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataroot", type=str)
    parser.add_argument("--train-cls", action='store_true')
    args = parser.parse_args()
    data_root = args.dataroot
    if args.train_cls:
        train_classifier(data_root)
    else:
        train_model(data_root)
