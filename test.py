import os
import argparse
import json
import joblib
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import cv2
import numpy as np
from addict import Dict
from PIL import Image
from utils.crf import DenseCRF
from utils.metric import scores
from dataloader.voc import VOCDataset
from models.msc import MSC
from models.vgg import Deeplab_LargeFOV
from models.resnet101 import DeepLabV2_Resnet101

def test(args):
    print('Start testing...')
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = range(torch.cuda.device_count())

    if args.model_type == 'vgg16':
        with open('./configs/vgg16.yaml', 'r') as f:
            CONFIG = Dict(yaml.load(f, Loader=yaml.FullLoader))
        model = Deeplab_LargeFOV(num_classes=CONFIG.DATASET.N_CLASSES)
        is_augment = True
        batch_size = 2
    elif args.model_type == 'resnet101':
        with open('./configs/resnet101.yaml', 'r') as f:
            CONFIG = Dict(yaml.load(f, Loader=yaml.FullLoader))
        model = MSC(base=DeepLabV2_Resnet101(layers=101, num_classes=CONFIG.DATASET.N_CLASSES, pretrained=False),
            scales=[0.5, 0.75],
        )
        is_augment = False
        batch_size = 1
    else:
        print('Unsupported model structure!')
        return
    
    logitPath = os.path.join(
        CONFIG.MODEL.NAME,
        'logitPath',
    )

    if not os.path.exists(logitPath):
        os.makedirs(logitPath)

    
    crop_size = 513
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model = model.to(device)

    val_loader = DataLoader(
        VOCDataset(dataRoot=args.root_path, split='val', crop_size=crop_size, \
            label_dir_path='SegmentationClass', is_augment=is_augment, is_scale=False, is_flip=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    for image_ids, images, gt_labels in tqdm(
        val_loader, total=len(val_loader), dynamic_ncols=True
    ):
        # Image
        images = images.to(device)

        # Forward propagation
        logits = model(images)

        for image_id, logit in zip(image_ids, logits):
            filename = os.path.join(logitPath, image_id + ".npy")
            np.save(filename, logit.cpu().numpy())
            
    print('Done.')


def crf(args):
    print('CRF post-processing...')
    torch.set_grad_enabled(False)
    
    if args.model_type == 'vgg16':
        with open('./configs/vgg16.yaml', 'r') as f:
            CONFIG = Dict(yaml.load(f, Loader=yaml.FullLoader))
    elif args.model_type == 'resnet101':
        with open('./configs/resnet101.yaml', 'r') as f:
            CONFIG = Dict(yaml.load(f, Loader=yaml.FullLoader))
    else:
        print('Unsupported model structure!')
        return

    crop_size = CONFIG.IMAGE.SIZE.TEST
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )

    # VOC color
    palette = []
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21] = np.array([[0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128]], dtype='uint8').flatten()
    
    logitPath = os.path.join(CONFIG.MODEL.NAME, 'logitPath')

    predictPath = os.path.join(CONFIG.MODEL.NAME, 'prediction')
    if not os.path.exists(predictPath):
        os.makedirs(predictPath)
    
    id_path = os.path.join('./datalists', 'val.txt')
    image_ids = [i.strip() for i in open(id_path) if not i.strip() == ' ']

    # Process per sample    
    def process(i):
        image_id = image_ids[i]
        gtPath = os.path.join(args.root_path, 'SegmentationClass', image_id + '.png')
        gt_label = Image.open(gtPath)
        imgPath = os.path.join(args.root_path, 'JPEGImages', image_id + '.jpg')
        raw_image = cv2.imread(imgPath, cv2.IMREAD_COLOR) # shape = [H, W, 3]
        H, W, _ = raw_image.shape
        if args.model_type == 'vgg16':
        ## padding raw_image to 513X513
            pad_h = max(crop_size - H, 0)
            pad_w = max(crop_size - W, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,
            }
            raw_image = cv2.copyMakeBorder(raw_image, value=[0, 0, 0], **pad_kwargs)
        raw_image = raw_image.astype(np.uint8)
        H, W, _ = raw_image.shape

        filename = os.path.join(logitPath, image_id + ".npy")
        logit = np.load(filename)
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=True)
        logit = F.softmax(logit, dim=1)[0].numpy()

        prob = postprocessor(raw_image, logit)
        label = np.argmax(prob, axis=0)

        # save predictions to path
        w, h = gt_label.size[0], gt_label.size[1]
        img_label = Image.fromarray(label.astype(np.uint8))
        img_label = img_label.crop((0, 0, w, h))
        label = np.asarray(img_label)
        gt_label = np.asarray(gt_label)
        img_label.putpalette(palette)
        img_label.save(os.path.join(predictPath, image_id + '.png'))

        return label, gt_label

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(image_ids))]
    )

    preds, gts = zip(*results)

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)
    print(score)
    savePath = os.path.join(CONFIG.MODEL.NAME, 'results')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    with open(savePath + '/results.json', "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='test', help='test model or crf post-processing')
    parser.add_argument('--root_path', default='/home/vious/data/VOCdevkit/VOC2012', help='path of PASCAL VOC2012 dataset')
    parser.add_argument('--model_type', default='', help='specify segmentation backbone (vgg16 | resnet101)')
    parser.add_argument('--model_path', default='', help='test model path')
    parser.add_argument('--label_path', default='./exp/labels', help='')
    parser.add_argument('--n_jobs', default=8, type=int, help='number of workers for crf post-processing')
    args = parser.parse_args()

    if args.type == 'test':
        test(args)
    elif args.type == 'crf':
        crf(args)