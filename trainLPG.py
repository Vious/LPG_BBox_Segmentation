import os
import torch
import torch.nn as nn
import numpy as np 
import time
import argparse
from dataloader.coco import COCOLabelForLPG
from models.hourglassnet import HourglassNet
from models.vgg import Deeplab_LargeFOV

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [int(id) for id in args.gpu_ids.split(',')]

    # create model
    Seg_model = Deeplab_LargeFOV(num_classes=61)
    Seg_model.load_state_dict(torch.load(args.init_seg_model_path))
    Seg_model = torch.nn.DataParallel(Seg_model, device_ids=device_ids)
    Seg_model.eval()
    Seg_model = Seg_model.to(device)

    LPG_model = HourglassNet({'fm':2}, 1, input_channel=5)
    LPG_model.load_state_dict(torch.load(args.init_lpg_model_path))
    LPG_model = torch.nn.DataParallel(LPG_model, device_ids=device_ids)
    LPG_model = LPG_model.to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(params=LPG_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    modelSavePath = os.path.join(args.model_save_path, 'Round_' + str(args.round))
    if not os.path.exists(modelSavePath):
        os.makedirs(modelSavePath)


    # dataset
    train_loader = torch.utils.data.DataLoader(
        COCOLabelForLPG(dataRoot=args.coco_root_dir, split='train2017', crop_size=args.coco_train_size, num_classes=2, is_scale=True, is_flip=True),
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4,
        drop_last = False,
    )
    
    CEL = nn.CrossEntropyLoss(ignore_index=255)

    # log file
    # log_file = open(args.log_path, 'w')
    loss_list, acc_list = [], []
    iters = 0
    while(True):
        for _, batch in enumerate(train_loader):
            _, images, labels, bbox_masks, cls_ids = batch
            images = images.to(device)
            labels = labels.to(device)
            bbox_masks = bbox_masks.to(device)
            cnn1_results = torch.ones([args.batch_size, 1, args.coco_train_size, args.coco_train_size]).to(device)

            for i in range(args.batch_size):
                with torch.no_grad():
                    cnn1_logits = Seg_model(images[i].view(1, 3, args.coco_train_size, args.coco_train_size)) # [1, 61, crop_size // 8, crop_size // 8]
                    cnn1_logits_up = nn.functional.interpolate(cnn1_logits, size=(args.coco_train_size, args.coco_train_size), mode='bilinear', align_corners=True) # [B, 61, crop_size, crop_size]
                    cnn1_logits_up_soft = nn.functional.softmax(cnn1_logits_up, dim=1) # [1, 61, crop_size, crop_size]
                cnn1_results[i, 0] = cnn1_logits_up_soft[0, int(cls_ids[i])]

            logits = LPG_model(torch.cat([cnn1_results, bbox_masks, images], dim=1)) # shape = [batch_size, 5, H, W]
            logits = logits[0]['fm']
            loss = CEL(logits, labels)
            # preds = torch.argmax(logits, dim=1)
            
            # acc = torch.eq(preds, labels).sum().float() / (labels.shape[0] * labels.shape[1] * labels.shape[2])
            loss_list.append(float(loss.cpu()))
            # acc_list.append(float(acc.cpu()))

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1
            if iters % args.num_print_log == 0:
                log_str = 'iters:{:6}, loss: {:6,.4f}'.format(iters, np.mean(loss_list))
                print(log_str)
                loss_list = []
                acc_list = []
            
            # save model and test model in val dataset
            if iters % args.num_save_model == 0:
                torch.save(LPG_model.module.state_dict(),  os.path.join(modelSavePath, str(iters) + '.pth'))
            
            if iters % args.num_update_lr == 0:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * (1 - float(iters) / args.num_max_iterations) ** 0.9
            
            if iters == args.num_max_iterations:
                exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', default='0', help='')
    
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=1.0e-4, type=float, help='')
    parser.add_argument('--momentum', default=0.9, type=float, help='')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='')
    parser.add_argument('--coco_train_size', default=512, type=int, help='')
    parser.add_argument('--num_print_log', default=50, type=int, help='')
    parser.add_argument('--num_update_lr', default=50, type=int, help='')
    parser.add_argument('--num_save_model', default=5000, type=int, help='')
    parser.add_argument('--num_max_iterations', default=100000, type=int, help='')
    parser.add_argument('--init_seg_model_path', default='', help='')
    parser.add_argument('--init_lpg_model_path', default='', help='')
    parser.add_argument('--model_save_path', default='./exp/lpg_models', help='path to save models')
    parser.add_argument('--round', default=2, type=int, help='number of training round')
    parser.add_argument('--log_path', default='./exp/log.txt', help='')
    # dataRoot 
    parser.add_argument('--coco_root_dir', default='/home/vious/data/cocoDataset/', help='')
        
    args = parser.parse_args()

    train(args)
