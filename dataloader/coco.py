import torch
import numpy as np 
import cv2 
import json
import os
import random
from torch.utils.data import Dataset

class COCOLabelForLPG(Dataset):
    def __init__(self, dataRoot, split='train2017', crop_size=321, num_classes=2, \
        is_scale=True, is_flip=True, down_sample=4):
        self.root_dir_path = dataRoot
        self.image_dir_path = os.path.join(self.root_dir_path, split)
        self.label_dir_path = os.path.join(self.root_dir_path, split + '_seg60')
        self.bbox_annos_path = os.path.join(self.root_dir_path, 'annotations' , 'coco_%s_bbox_annos.json' % split)
        with open(self.bbox_annos_path, 'r') as f:
            self.bbox_anno_dict = json.load(f)
        
        coco_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        voc_ids = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
        valid_ids = list(set(coco_ids) - set(voc_ids)) # get ids in coco not in voc
        self.cat60_ids = {v: i+1 for i, v in enumerate(valid_ids)} # map ids to 1 - 60

        self.indexs = []
        for image_id, bbox_annos in self.bbox_anno_dict.items():
            bbox_annos = np.array(bbox_annos)
            cls_ids = list(set(bbox_annos[:, 4]))
            for cls_id in cls_ids:
                if int(cls_id) in valid_ids:
                    self.indexs.append([image_id, int(cls_id)])

        self.mean_bgr = np.array((104.008, 116.669, 122.675))
        self.split = split
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.down_sample = down_sample
        self.ignore_label = 0
        self.base_size = None
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        self.is_scale = is_scale
        self.is_flip = is_flip
    
    def __len__(self):
        return len(self.indexs)
    
    def __getitem__(self, index):
        image_id, cls_id = self.indexs[index]
        image_path = os.path.join(self.image_dir_path, str(int(image_id)).zfill(12) + '.jpg')
        label_path = os.path.join(self.label_dir_path, str(int(image_id)).zfill(12) + '.png')
        bbox_annos = self.bbox_anno_dict[image_id]
        cls_id60 = self.cat60_ids[cls_id]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label_seg = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
        bbox_mask = np.zeros_like(label_seg).astype(np.float32)
        anno_list = []
        for bbox in bbox_annos:
            if cls_id == bbox[4]:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                bbox_mask[y1:(y2+1), x1:(x2+1)] = 1.0
                anno_list.append(bbox)
        annos = np.array(anno_list) # bounding box annotations

        # put segmentation label to 0 and 1
        label = np.zeros_like(label_seg).astype(np.int32)
        label[label_seg == cls_id60] = 1

        # Scaling
        if self.is_scale:
            h, w = label.shape
            if self.base_size:
                if h > w:
                    h, w = (self.base_size, int(self.base_size * w / h))
                else:
                    h, w = (int(self.base_size * h / w), self.base_size)
            scale_factor = random.choice(self.scales)
            h, w = (int(h * scale_factor), int(w * scale_factor))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            bbox_mask = cv2.resize(bbox_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            annos[:, :4] = annos[:, :4] * scale_factor
        
        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)
            bbox_mask = cv2.copyMakeBorder(bbox_mask, value=0, **pad_kwargs)
        
        # get bounding box
        min_h, max_h = int(min(annos[:, 1])), int(max(annos[:, 3]))
        min_w, max_w = int(min(annos[:, 0])), int(max(annos[:, 2]))
        # print(min_h, min_w, max_h, max_w)

        # Cropping
        h, w = label.shape
        start_h = random.randint(max(0, min_h - self.crop_size), min(max_h, h - self.crop_size))
        start_w = random.randint(max(0, min_w - self.crop_size), min(max_w, w - self.crop_size))
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        bbox_mask = bbox_mask[start_h:end_h, start_w:end_w]
        annos[:, [0, 2]] = annos[:, [0, 2]] - start_w
        annos[:, [1, 3]] = annos[:, [1, 3]] - start_h
        # print(annos)

        if self.is_flip:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
                bbox_mask = np.fliplr(bbox_mask).copy() # HW
                annos[:, [0, 2]] = self.crop_size - annos[:, [2, 0]]
        
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        bbox_mask = bbox_mask[np.newaxis, :, :] # [1, H, W]

        if self.down_sample == 8:
            label = cv2.resize(label, (self.crop_size // self.down_sample, self.crop_size // self.down_sample), interpolation=cv2.INTER_NEAREST)
        else:
            label = cv2.resize(label, (self.crop_size // self.down_sample, self.crop_size // self.down_sample), interpolation=cv2.INTER_NEAREST)

        return str(int(image_id)).zfill(12), image.astype(np.float32), label.astype(np.int64), bbox_mask.astype(np.float32), str(cls_id60)



class COCOBBoxForGen(Dataset):
    def __init__(self, dataRoot, split='train2017', crop_size=321, num_classes=61, is_augment=True, is_scale=True, is_flip=True):
        self.root = dataRoot
        # self.ann_dir_path = os.path.join(self.root, 'annotations2017')
        self.image_dir_path = os.path.join(self.root, split)
        self.label_dir_path = os.path.join(self.root, split + '_seg60')
        self.anno_path = os.path.join(self.root, 'annotations' , 'coco_%s_bbox_annos.json' % split)
        with open(self.anno_path, 'r') as f:
            self.anno_dict = json.load(f)

        self.image_ids = [i.split('.')[0] for i in os.listdir(self.label_dir_path)]

        self._coco_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._voc_ids = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
        self._valid_ids = list(set(self._coco_ids) - set(self._voc_ids))
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        self.mean_bgr = np.array((104.008, 116.669, 122.675)) # voc=(104.008, 116.669, 122.675), coco=
        self.split = split
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.down_sample = 4
        self.ignore_label = 255
        self.base_size = None
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        self.is_augment = is_augment
        self.is_scale = is_scale
        self.is_flip = is_flip

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_path = os.path.join(self.image_dir_path, image_id + '.jpg')

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        # width, height = image.shape[1], image.shape[0]
        
        anno_list = self.anno_dict[str(int(image_id))]
        anno_list = [anno for anno in anno_list if anno[4] in self._valid_ids]
        annos = np.array(anno_list).astype(np.float32)
        if self.is_augment:
            image, annos = self._augmentation(image, annos)
        
        anno_list = []
        num = 0
        for anno in annos:
            if self._get_iou(anno, self.crop_size, self.crop_size) > 0:
                anno_list.append([anno[0], anno[1], anno[2], anno[3], self.cat_ids[anno[4]] + 1])
                num += 1
        for i in range(100 - num):
            anno_list.append([-1, -1, -1, -1, -1])
        annos = np.array(anno_list)

        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)

        return image_id, image.astype(np.float32), annos.astype(np.int64)

    def _get_iou(self, anno, width, height):
        if anno[3] - anno[1] <= 0 or anno[2] - anno[0] <= 0:
            return -1
        W = min(anno[2], width) - max(anno[0], 0)
        H = min(anno[3], height) - max(anno[1], 0)
        if W <= 0 or H <= 0:
            return 0
        SA = (anno[2] - anno[0]) * (anno[3] - anno[1])
        SB = width * height
        cross = W * H
        return cross / (SA + SB - cross)

    def _augmentation(self, image, annos):
        # Scaling
        if self.split.__contains__('train') and self.is_scale:
            h, w, _ = image.shape
            if self.base_size:
                if h > w:
                    h, w = (self.base_size, int(self.base_size * w / h))
                else:
                    h, w = (int(self.base_size * h / w), self.base_size)
            scale_factor = random.choice(self.scales)
            h, w = (int(h * scale_factor), int(w * scale_factor))
            annos[:, :4] = annos[:, :4] * scale_factor
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Padding to fit for crop_size
        h, w, _ = image.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)

        # Cropping
        h, w, _ = image.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        annos[:, [0, 2]] = annos[:, [0, 2]] - start_w
        annos[:, [1, 3]] = annos[:, [1, 3]] - start_h

        if self.split.__contains__('train') and self.is_flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                
                annos[:, [0, 2]] = self.crop_size - annos[:, [2, 0]]
                # annos[:, [1, 3]] = self.crop_size - annos[:, [3, 1]]
        
        return image, annos