import os
import torch
import torch.nn as nn
import numpy as np 
import cv2
import argparse
from PIL import Image
from dataloader.voc import VOCBBoxForGen
from dataloader.coco import COCOBBoxForGen
from models.hourglassnet import HourglassNet
from models.vgg import Deeplab_LargeFOV
from models.msc import MSC
from models.resnet101 import DeepLabV2_Resnet101

def generate_voc_proposas(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [int(id) for id in args.gpu_ids.split(',')]

    # create model
    if args.model_type == 'vgg16':
        seg_model = Deeplab_LargeFOV(num_classes=21)
    elif args.model_type == 'resnet101':
        seg_model = MSC(base=DeepLabV2_Resnet101(layers=101, num_classes=21, pretrained=True),
            scales=[0.5, 0.75],
        )
    
    seg_model.load_state_dict(torch.load(args.init_voc_model_path))
    seg_model = torch.nn.DataParallel(seg_model, device_ids=device_ids)
    seg_model.eval()
    seg_model = seg_model.to(device)

    if args.round == 1:
        LPG_model = HourglassNet({'fm':2}, 1, input_channel=4)
    else:
        LPG_model = HourglassNet({'fm':2}, 1, input_channel=5)
    LPG_model.load_state_dict(torch.load(args.model_path_test))
    LPG_model = torch.nn.DataParallel(LPG_model, device_ids=device_ids)
    LPG_model.eval()
    LPG_model = LPG_model.to(device)

    # dataset
    voc_loader = torch.utils.data.DataLoader(
        VOCBBoxForGen(args.voc_root_dir, split=args.voc_split, crop_size=args.voc_test_size, is_scale=False, is_flip=False),
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        drop_last = False
    )

    labelSavePath = os.path.join(args.voc_proposals_dir, args.model_type + '_Round' + str(args.round))
    if not os.path.exists(labelSavePath):
        os.makedirs(labelSavePath)

    # voc color
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

    # iterations
    for iter_id, batch in enumerate(voc_loader):
        image_ids, images, annos = batch
        images = images.to(device)

        if iter_id % 1000 == 0:
            print("generate %s images" % iter_id)

        if args.round != 1:
            with torch.no_grad():
                cnn1_logits = seg_model(images) # [B, 61, crop_size // 8, crop_size // 8]
                cnn1_logits_up = nn.functional.interpolate(cnn1_logits, size=(args.voc_test_size, args.voc_test_size), mode='bilinear', align_corners=True) # [B, 61, crop_size, crop_size]
                cnn1_logits_up_soft = nn.functional.softmax(cnn1_logits_up, dim=1) # [B, 61, crop_size, crop_size]

        for i in range(annos.shape[0]):
            img = cv2.imread(args.voc_root_dir + 'JPEGImages/%s.jpg' % (image_ids[i]))
            height, width = img.shape[0], img.shape[1]

            cls_id_list = []
            for j in range(annos.shape[1]):
                if annos[i][j][4] == -1:
                    continue
                if not cls_id_list.__contains__(int(annos[i][j][4])):
                    cls_id_list.append(int(annos[i][j][4]))
            
            pred_label = torch.zeros([args.voc_test_size, args.voc_test_size]).long().to(device)
            for cls_id in cls_id_list:
                bbox_mask_ = torch.autograd.Variable(torch.zeros(args.voc_test_size, args.voc_test_size), requires_grad = False).to(device) # shape=[H, W]
                for j in range(annos.shape[1]):
                    if annos[i][j][4] == -1 or int(annos[i][j][4]) != cls_id:
                        continue
                    x1, y1, x2, y2 = int(annos[i][j][0]), int(annos[i][j][1]), int(annos[i][j][2]), int(annos[i][j][3])
                    bbox_mask_[y1:(y2+1), x1:(x2+1)] = 1.0
                bbox_mask = bbox_mask_.view(1, args.voc_test_size, args.voc_test_size)
                
                if args.round != 1:
                    cnn1_results = torch.ones([1, args.voc_test_size, args.voc_test_size]).to(device)
                    cnn1_results[0] = cnn1_logits_up_soft[0, cls_id]
                    inputs = torch.cat((cnn1_results, bbox_mask, images[i]), dim=0).view(1, 5, args.voc_test_size, args.voc_test_size)
                else:
                    inputs = torch.cat(( bbox_mask, images[i]), dim=0).view(1, 4, args.voc_test_size, args.voc_test_size)

                with torch.no_grad():
                    logits = LPG_model(inputs)
                    logits = logits[0]['fm']
                    logits_up = nn.functional.interpolate(logits, size=(args.voc_test_size, args.voc_test_size), mode='bilinear', align_corners=True)
                pred = torch.argmax(logits_up, dim=1)[0] * cls_id * bbox_mask[0]
                pred_label = torch.where(pred_label == 0, pred.long(), pred_label)

            # save label
            img_label = pred_label.cpu().numpy().astype(np.uint8)
            img_label = Image.fromarray(img_label)
            img_label = img_label.crop((0, 0, width, height))
            img_label.putpalette(palette)
            img_label.save( os.path.join(labelSavePath, image_ids[i] + '.png') )

    print('Done.')


def generate_coco_proposas(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [int(id) for id in args.gpu_ids.split(',')]

    # create model
    seg_model = Deeplab_LargeFOV(num_classes=61)
    seg_model.load_state_dict(torch.load(args.init_coco_model_path))
    seg_model = torch.nn.DataParallel(seg_model, device_ids=device_ids)
    seg_model.eval()
    seg_model = seg_model.to(device)

    if args.round == 1:
        LPG_model = HourglassNet({'fm':2}, 1, input_channel=4)
    else:
        LPG_model = HourglassNet({'fm':2}, 1, input_channel=5)
    LPG_model.load_state_dict(torch.load(args.model_path_test))
    LPG_model.eval()
    LPG_model = LPG_model.to(device)

    # dataset
    coco_loader = torch.utils.data.DataLoader(
        COCOBBoxForGen(args.coco_root_dir, split='train2017', crop_size=args.coco_test_size, is_scale=False, is_flip=False),
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        drop_last = False
    )

    labelSavePath = os.path.join(args.coco_proposals_dir, args.model_type + '_Round' + str(args.round))
    if not os.path.exists(labelSavePath):
        os.makedirs(labelSavePath)

    # iterations
    for iter_id, batch in enumerate(coco_loader):
        image_ids, images, annos = batch
        images = images.to(device)

        if iter_id % 10 == 0:
            print('%s indexed' % iter_id)

        if args.round != 1:
            with torch.no_grad():
                cnn1_logits = seg_model(images) # [B, 61, crop_size // 8, crop_size // 8]
                cnn1_logits_up = nn.functional.interpolate(cnn1_logits, size=(args.coco_test_size, args.coco_test_size), mode='bilinear', align_corners=True) # [B, 61, crop_size, crop_size]
                cnn1_logits_up_soft = nn.functional.softmax(cnn1_logits_up, dim=1) # [B, 61, crop_size, crop_size]

        for i in range(annos.shape[0]):
            cls_id_list = []
            for j in range(annos.shape[1]):
                if annos[i][j][4] == -1:
                    continue
                if not cls_id_list.__contains__(int(annos[i][j][4])):
                    cls_id_list.append(int(annos[i][j][4]))
            
            pred_label = torch.zeros([args.coco_test_size, args.coco_test_size]).long().to(device)
            for cls_id in cls_id_list:
                bbox_mask = torch.autograd.Variable(torch.zeros(args.coco_test_size, args.coco_test_size), requires_grad = False).to(device) # shape=[H, W]
                for j in range(annos.shape[1]):
                    if annos[i][j][4] == -1 or int(annos[i][j][4]) != cls_id:
                        continue
                    x1, y1, x2, y2 = int(annos[i][j][0]), int(annos[i][j][1]), int(annos[i][j][2]), int(annos[i][j][3])
                    bbox_mask[y1:(y2+1), x1:(x2+1)] = 1.0
                bbox_mask = bbox_mask.view(1, args.coco_test_size, args.coco_test_size)
                
                if args.round != 1:
                    cnn1_results = torch.ones([1, args.coco_test_size, args.coco_test_size]).to(device)
                    cnn1_results[0] = cnn1_logits_up_soft[0, cls_id]
                    inputs = torch.cat((cnn1_results, bbox_mask, images[i]), dim=0).view(1, 5, args.coco_test_size, args.coco_test_size)
                else:
                    inputs = torch.cat(( bbox_mask, images[i]), dim=0).view(1, 4, args.coco_test_size, args.coco_test_size)
                with torch.no_grad():
                    logits = LPG_model(inputs)
                    logits = logits[0]['fm']
                    logits_up = nn.functional.interpolate(logits, size=(args.coco_test_size, args.coco_test_size), mode='bilinear', align_corners=True)
                pred = torch.argmax(logits_up, dim=1)[0] * cls_id * bbox_mask[0]
                pred_label = torch.where(pred_label == 0, pred.long(), pred_label)

            # save label
            img = cv2.imread(args.coco_root_dir + 'train2017/%s.jpg' % (image_ids[i]))
            height, width = img.shape[0], img.shape[1]
            img_label = pred_label.cpu().numpy().astype(np.uint8)
            img_label = Image.fromarray(img_label)
            img_label = img_label.crop((0, 0, width, height))
            img_label.save(os.path.join(labelSavePath, image_ids[i] + '.png'))

    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='voc', help='which dataset to gen pseudo labels (voc | coco)')
    parser.add_argument('--model_type', default='vgg16', help='Specify segmentation backbone (vgg16 | resnet101)')
    parser.add_argument('--round', default=1, type=int, help='set number of round')
    parser.add_argument('--gpu_ids', default='0', help='')
        
    # for generate voc proposals
    parser.add_argument('--voc_root_dir', default='/home/vious/data/VOCdevkit/VOC2012/', help='')
    parser.add_argument('--init_voc_model_path', default='', help='segmentation backbone for PASCAL VOC dataset')
    parser.add_argument('--voc_test_size', default=512, type=int, help='')
    parser.add_argument('--voc_split', default='train_aug', help='')
    parser.add_argument('--voc_proposals_dir', default='./exp/pseudo_labels/', help='')

    # for generate coco proposals
    parser.add_argument('--init_coco_model_path', default='', help='')
    parser.add_argument('--coco_root_dir', default='/home/vious/data/cocoDataset/', help='')
    parser.add_argument('--coco_test_size', default=640, type=int, help='')
    parser.add_argument('--coco_proposals_dir', default='./exp/coco_labels/', help='')

    # for load LPG model
    parser.add_argument('--model_path_test', default='', help='pretrained LPG model path')
    args = parser.parse_args()

    if args.dataset == 'voc':
        generate_voc_proposas(args)
    elif args.dataset == 'coco':
        generate_coco_proposas(args)
    else:
        print('None')
