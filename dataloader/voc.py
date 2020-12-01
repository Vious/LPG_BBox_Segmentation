import os
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
import random
from PIL import Image
from torch.utils.data import Dataset

### original pascal voc 2012 datset (1464 training images)
class VOC2012(Dataset):
    def __init__(self, dataRoot, split, ignore_label, mean_bgr, augment=True, \
        base_size=None, crop_size=321, scales=(1.0), flip=True):
        super(VOC2012, self).__init__()

        self.dataRoot = dataRoot
        self.split = split
        self.ignore_label = ignore_label
        self.mean_bgr = np.array(mean_bgr)
        self.augment = augment
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.files = []
        self._set_files()

        cv2.setNumThreads(0)


    def _set_files(self):
        self.fileDir = os.path.join(self.dataRoot, 'VOC2012')
        self.imageDir = os.path.join(self.fileDir, 'JPEGImages')
        self.labelDir = os.path.join(self.fileDir, 'SegmentationClass')

        if self.split in ["train", "trainval", "val", "test"]:
            file_list = os.path.join(
                'datalists', self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _augmentation(self, image, label):
        # Scaling
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


        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW

        return image, label


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        imgId = self.files[index]
        imgPath = os.path.join(self.dataRoot, self.imageDir, imgId + '.jpg')
        labelPath = os.path.join(self.dataRoot, self.labelDir, imgId + '.png')
        
        image = cv2.imread(imgPath, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(labelPath), dtype=np.int32)

        if self.augment:
            image, label = self._augmentation(image, label)
        
        image -= self.mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return imgId, image.astype(np.float32), label.astype(np.int64)


### VOC augmented dataset (10582 training images)
class VOCAug(Dataset):
    def __init__(self, dataRoot, split, ignore_label, mean_bgr, augment=True, \
        base_size=None, crop_size=321, scales=(1.0), flip=True):
        super(VOCAug, self).__init__()

        self.dataRoot = dataRoot
        self.split = split
        self.ignore_label = ignore_label
        self.mean_bgr = np.array(mean_bgr)
        self.augment = augment
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.files = []
        self._set_files()

        cv2.setNumThreads(0)
    
    def _set_files(self):
        self.fileDir = os.path.join(self.dataRoot, 'VOC2012')
        self.imageDir = os.path.join(self.fileDir, 'JPEGImages')
        self.labelDir = os.path.join(self.fileDir, 'labels_voc_r3_resnet_semi')

        if self.split in ["train_aug", "val", "test"]:
            file_list = os.path.join(
                'datalists', self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))
    

    def _augmentation(self, image, label):
        # Scaling
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


        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW

        return image, label


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        imgId = self.files[index]
        imgPath = os.path.join(self.dataRoot, self.imageDir, imgId + '.jpg')
        labelPath = os.path.join(self.dataRoot, self.labelDir, imgId + '.png')
        
        image = cv2.imread(imgPath, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(labelPath), dtype=np.int32)

        if self.augment:
            image, label = self._augmentation(image, label)
        
        image -= self.mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return imgId, image.astype(np.float32), label.astype(np.int64)

    


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    batch_size = 32

    # set dataRoot = your VOCdevkit path
    dataset = VOCAug(
        dataRoot='/opt/data/private/chaohaoData/VOCdevkit',
        split="train_aug",
        ignore_label=255,
        mean_bgr=(104.008, 116.669, 122.675),
        augment=True,
        base_size=None,
        crop_size=321,
        scales=(0.5, 0.75, 1.0, 1.25, 1.5),
        flip=True,
    )
    print(dataset)

    loader = DataLoader(dataset, batch_size=batch_size)
    count = 0
    for i, (image_ids, images, labels) in enumerate(loader):
        print(count, 'len is :', len(image_ids))
        print(images.size())
        print(labels.size())
        count += 1