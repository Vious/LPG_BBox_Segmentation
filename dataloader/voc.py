import os
import cv2
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import utils.pascal as pascal

### VOC augmented dataset (10582 training images)
class VOCDataset(Dataset):
    def __init__(self, dataRoot='/home/vious/data/VOCdevkit/VOC2012', \
        split='train_aug', crop_size=321, label_dir_path='labels_voc_r1_vgg', \
            is_augment=True, is_scale=True, is_flip=True):
        
        print('Current dataset is: ', label_dir_path)
        self.root = dataRoot
        self.ann_dir_path = os.path.join(self.root, 'Annotations')
        self.image_dir_path = os.path.join(self.root, 'JPEGImages')
        self.label_dir_path = os.path.join(self.root, label_dir_path) # SegmentationClassAug_Round1
        self.id_path = os.path.join('./datalists', split + '.txt')

        self.image_ids = [i.strip() for i in open(self.id_path) if not i.strip() == ' ']
        print('%s datasets num = %s' % (split, self.__len__()))

        self.mean_bgr = np.array((104.008, 116.669, 122.675))
        self.split = split
        self.crop_size = crop_size
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
        label_path = os.path.join(self.label_dir_path, image_id + '.png')
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)

        if self.is_augment:
            image, label = self._augmentation(image, label)
        
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        return image_id, image.astype(np.float32), label.astype(np.int64)
    
    def _augmentation(self, image, label):
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
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int64)

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

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        # print(bbox)

        if self.is_flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label
    

class VOCBBoxForGen(torch.utils.data.Dataset):
    def __init__(self, dataroot, split='train_aug', crop_size=321, is_augment=True, is_scale=True, is_flip=True):
        self.root = dataroot
        self.ann_dir_path = os.path.join(self.root, 'Annotations')
        self.image_dir_path = os.path.join(self.root, 'JPEGImages')
        self.id_path = os.path.join('./datalists', split + '.txt')

        self.image_ids = [i.strip() for i in open(self.id_path) if not i.strip() == ' ']
        print('%s datasets num = %s' % (split, self.__len__()))

        self.mean_bgr = np.array((104.008, 116.669, 122.675))
        self.split = split
        self.crop_size = crop_size
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
        anno_path = os.path.join(self.ann_dir_path, image_id + '.xml')
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        annos = np.array(pascal.parse_xml(anno_path)).astype(np.float32)

        if self.is_augment:
            image, annos = self._augmentation(image, annos)

        anno_list = []
        num = 0
        for anno in annos:
            if self._get_iou(anno, self.crop_size, self.crop_size) > 0:
                anno_list.append([anno[0], anno[1], anno[2], anno[3], int(anno[4])])
                num += 1
        for i in range(100 - num):
            anno_list.append([-1, -1, -1, -1, -1])
        annos = np.array(anno_list) # shape = [100, 5]
    
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        return image_id, image.astype(np.float32), annos.astype(np.int64)
    
    def _augmentation(self, image, annos):
        # Scaling
        if self.is_scale:
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
        # print(bbox)

        if self.is_flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                annos[:, [0, 2]] = self.crop_size - annos[:, [2, 0]]
                # bbox[:, [1, 3]] = self.crop_size - bbox[:, [1, 3]]
        
        return image, annos

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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    batch_size = 30

    # set dataRoot = your VOCdevkit path, label_path is the (Pseudo) labels you use
    dataset = VOCDataset(
        dataRoot='/home/vious/data/VOCdevkit/VOC2012',
        split="train_aug",
        ignore_label=255,
        label_Path= 'labels_voc_r3_resnet',
        crop_size=321,
        is_augment=True,
        is_scale=True,
        is_flip=True,
    )
    print(dataset)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    count = 0
    for i, (image_ids, images, labels) in enumerate(loader):
        print('Iteration:', count, 'len is :', len(image_ids))
        print(images.size())
        print(labels.size())
        count += 1