import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from addict import Dict
from PIL import Image
from dataloader.voc import VOCDataset
from models.vgg import Deeplab_LargeFOV
from models.resnet101 import DeepLabV2_Resnet101
from models.msc import MSC
from utils.losses import build_metrics

def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device

def get_params(model, key, modelName = 'vgg16'):
    if modelName == 'vgg16':
        if key == '1x':
            for m in model.named_modules():
                if isinstance(m[1], nn.Conv2d):
                    if m[0] != 'features.38':
                        yield m[1].weight
        if key == '2x':
            for m in model.named_modules():
                if isinstance(m[1], nn.Conv2d):
                    if m[0] != 'features.38':
                        yield m[1].bias
        if key == '10x':
            for m in model.named_modules():
                if isinstance(m[1], nn.Conv2d):
                    if m[0] == 'features.38':
                        yield m[1].weight
        if key == '20x':
            for m in model.named_modules():
                if isinstance(m[1], nn.Conv2d):
                    if m[0] == 'features.38':
                        yield m[1].bias
    elif modelName == 'resnet101':
        # For convolutional layers before last layer
        if key == "1x":
            for m in model.named_modules():
                if "layer" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p
        # For conv weight in the ASPP module
        if key == "10x":
            for m in model.named_modules():
                if "asppModule" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        yield m[1].weight
        # For conv bias in the ASPP module
        if key == "20x":
            for m in model.named_modules():
                if "aspp" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        yield m[1].bias


def trainVGG16(args):
    with open('./configs/vgg16.yaml', 'r') as f:
        CONFIG = Dict(yaml.load(f, Loader=yaml.FullLoader))
    
    device = get_device(args.cuda)
    torch.backends.cudnn.benchmark = True
    model = Deeplab_LargeFOV(num_classes=CONFIG.DATASET.N_CLASSES)
    ## model.load_state_dict(torch.load(CONFIG.MODEL.INIT_MODEL)) # Just for first round
    ## specify a pretrained model for later training round
    model.load_state_dict(torch.load(args.init_model_path)) 
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.to(device)

    ## set optimizer
    optimizer = torch.optim.SGD(
        params = [
            {
                'params': get_params(model, '1x', modelName = 'vgg16'),
                'lr': CONFIG.SOLVER.LR,
                'weight_decay': CONFIG.SOLVER.WEIGHT_DECAY
            },
            {
                'params': get_params(model, '2x', modelName = 'vgg16'),
                'lr': CONFIG.SOLVER.LR * 2,
                'weight_decay': 0
            },
            {
                'params': get_params(model, '10x'),
                'lr': CONFIG.SOLVER.LR * 10,
                'weight_decay': CONFIG.SOLVER.WEIGHT_DECAY
            },
            {
                'params': get_params(model, '20x'),
                'lr': CONFIG.SOLVER.LR * 20,
                'weight_decay': 0
            },
        ],
        momentum = 0.9,
    )

    ## model save path
    if args.save_path != '':
        pathSave = args.save_path
    else:
        pathSave = os.path.join(CONFIG.MODEL.NAME, 'checkpoints')
    
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)

    print('Set data...')
    train_loader = torch.utils.data.DataLoader(
        VOCDataset(dataRoot=args.root_path, split=CONFIG.DATASET.SPLIT.TRAIN, \
            label_dir_path=args.label_path_name, crop_size=321, is_augment=True, is_scale=True, is_flip=True), 
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        shuffle=True,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        drop_last=True
    )

    # Learning rate policy
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    print('Start to train...')
    iters = 0
    # log_file = open(log_path, 'w')
    loss_iters, accuracy_iters = [], []
    for epoch in range(1, 4000):
        model.train()
        for iter_id, batch in enumerate(train_loader):
            loss_seg = build_metrics(model, batch, device)
            optimizer.zero_grad()
            loss_seg.backward()
            optimizer.step()

            loss_iters.append(float(loss_seg.cpu()))
            # accuracy_iters.append(float(accuracy))

            iters += 1
            if iters % CONFIG.SOLVER.ITER_TB == 0:
                log_str = 'iters:{:4}, loss:{:6,.4f}'.format(iters, np.mean(loss_iters))
                print(log_str)
                loss_iters = []
            
            if iters % CONFIG.SOLVER.ITER_SAVE == 0:
                torch.save(model.module.state_dict(), os.path.join(pathSave, 'VGG16_' + str(iters) + '.pth'))

            # poly
            for group in optimizer.param_groups:
                group["lr"] = group['initial_lr'] * (1 - float(iters) / CONFIG.SOLVER.ITER_MAX) ** 0.9

            if iters == CONFIG.SOLVER.ITER_MAX:
                exit()


def trainResNet101(args):
    with open('./configs/resnet101.yaml', 'r') as f:
        CONFIG = Dict(yaml.load(f, Loader=yaml.FullLoader))
    
    device = get_device(args.cuda)
    torch.backends.cudnn.benchmark = True
    ## define model
    model = MSC(base=DeepLabV2_Resnet101(layers=101, num_classes=CONFIG.DATASET.N_CLASSES, pretrained=True),
        scales=[0.5, 0.75],
    )
    ## specify a pretrained model for later training round
    model.load_state_dict(torch.load(args.init_model_path)) ## comment this line if training for the first round 
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.to(device)

    ## set optimizer
    optimizer = torch.optim.SGD(
        params=[
            {
                "params": get_params(model.module, key="1x", modelName='resnet101'),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x", modelName='resnet101'),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x", modelName='resnet101'),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum = CONFIG.SOLVER.MOMENTUM,
    )

    ## model save path
    if args.save_path != '':
        pathSave = args.save_path
    else:
        pathSave = os.path.join(CONFIG.MODEL.NAME, 'checkpoints')
    
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)

    print('Set data...')
    train_loader = torch.utils.data.DataLoader(
        VOCDataset(dataRoot=args.root_path, split=CONFIG.DATASET.SPLIT.TRAIN, \
            crop_size=321, is_augment=True, is_scale=True, is_flip=True), 
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        shuffle=True,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
    )

    loader_iter = iter(train_loader)

    # Learning rate policy
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    ## uncomment this line for finetuning
    # model.module.base.freeze_bn()

    ## define cross entropy loss
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion = nn.DataParallel(criterion)
    criterion.to(device)

    print('Start to train...')
    iters = 0
    loss_iters, accuracy_iters = [], []
    for epoch in range(1, CONFIG.SOLVER.ITER_MAX * 2):
        model.train()

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                _, images, labels = next(loader_iter)
            except:
                loader_iter = iter(train_loader)
                _, images, labels = next(loader_iter)

            # Propagate forward
            logits = model(images.to(device))
            # Loss
            iter_loss = 0
            
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, H, W = logit.shape
                labels_ = resize_labels(labels, size=(H, W))
                iter_loss += criterion(logit, labels_.to(device)).mean()

            # Propagate backward (just compute gradients wrt the loss)
            iter_loss /= CONFIG.SOLVER.ITER_SIZE
            iter_loss.backward()

        loss_iters.append(float(iter_loss.cpu()))        
        optimizer.step()

        iters += 1
        if iters % CONFIG.SOLVER.ITER_TB == 0:
            log_str = 'iters:{:4}, loss:{:6,.4f}'.format(iters, np.mean(loss_iters))
            print(log_str)
            loss_iters = []
        
        if iters % CONFIG.SOLVER.ITER_SAVE == 0:
            torch.save(model.module.state_dict(), os.path.join(pathSave, 'ResNet101_' + str(iters) + '.pth'))

        # poly
        for group in optimizer.param_groups:
            group["lr"] = group['initial_lr'] * (1 - float(iters) / CONFIG.SOLVER.ITER_MAX) ** 0.9

        if iters == CONFIG.SOLVER.ITER_MAX:
            exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/home/vious/data/VOCdevkit/VOC2012', help='path of PASCAL VOC2012 dataset')
    parser.add_argument('--label_path_name', default='labels_voc_r1_vgg', help='path name of generated pseudo labels')
    parser.add_argument('--model_type', default='', help='Specify segmentation backbone (vgg16 | resnet101)')
    parser.add_argument('--init_model_path', default='', help='pretrained model path')
    parser.add_argument('--save_path', default='', help='path to save models')
    parser.add_argument('--cuda', default=True, type=bool, help='use cuda or not')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for dataloader')

    args = parser.parse_args()

    if args.model_type == 'vgg16':
        print('Start training segmentation backbone DeepLab-LargeFOV of VGG16...')
        trainVGG16(args)
    elif args.model_type == 'resnet101':
        print('Start training segmentation backbone DeepLab-ResNet101...')
        trainResNet101(args)
    else:
        print('Unsupported network now!')
