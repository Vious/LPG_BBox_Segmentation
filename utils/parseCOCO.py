import os
import cv2
import json
import numpy as np
from pycocotools.coco import COCO


def parseCOCOBBox(dataRoot, split='train2017'):
    annotationPath = os.path.join(dataRoot, 'annotations', 'instances_%s.json' % split)
    print(annotationPath)

    with open(annotationPath) as f:
        annos = json.load(f)
    
    dataDict = {}
    for ann in annos['annotations']:
        imageId = ann['image_id']
        catId = ann['category_id']
        bbox = ann['bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        
        if not dataDict.keys().__contains__(imageId):
            dataDict[imageId] = [bbox + [catId]]
        else:
            dataDict[imageId].append(bbox + [catId])

    fileSavePath = os.path.join(dataRoot, 'annotations', \
        'coco_%s_bbox_annos.json' % split)

    print('Write done bbox annotations...')
    print('Save to', fileSavePath)
    with open(fileSavePath, 'w') as f:
        json.dump(dataDict, f)

    print('Done.')

def genBboxLabel(dataRoot, split='train2017'):
    COCO_Ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
        82, 84, 85, 86, 87, 88, 89, 90
        ]

    voc_ids = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
    new_ids = list(set(COCO_Ids) - set(voc_ids))
    print('Length of used classed is {:d}'.format(len(new_ids)))
    print(new_ids)
    
    valid_ids = {v : i + 1 for i, v in enumerate(new_ids)}
    print(valid_ids)

    boxLabelSavePath = os.path.join(dataRoot, '%s_bbox60' % split)
    if not os.path.exists(boxLabelSavePath):
        os.makedirs(boxLabelSavePath)

    bboxAnnosPath = os.path.join(dataRoot, 'annotations', 'coco_%s_bbox_annos.json' % split)
    with open(bboxAnnosPath, 'r') as f:
        bboxAnnos = json.load(f)
    
    count = 0

    for imageId, bboxes in bboxAnnos.items():
        if count % 1000 == 0:
            print('Processing {:d}\'s bbox label'.format(count))
        catIds = []
        for bbox in bboxes:
            if not catIds.__contains__(bbox[4]):
                catIds.append(bbox[4])
        
        if set(voc_ids) > set(catIds):
            continue
        
        imageName = os.path.join(dataRoot, split, str(int(imageId)).zfill(12) + '.jpg')
        image = cv2.imread(imageName)
        bboxMask = np.zeros(shape=[image.shape[0], image.shape[1]], dtype=np.uint8)

        for bbox in bboxes:
            if not voc_ids.__contains__(bbox[4]):
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                bboxMask[y1:(y2+1), x1:(x2+1)] = valid_ids[int(bbox[4])]

        cv2.imwrite(os.path.join(boxLabelSavePath, str(int(imageId)).zfill(12) + '.png'), bboxMask)
        count += 1

    print('Done.')


def parseCOCOLabel(dataRoot, split='train2017'):
    COCO_Ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
        82, 84, 85, 86, 87, 88, 89, 90
        ]

    voc_ids = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
    new_ids = list(set(COCO_Ids) - set(voc_ids))
    print('Length of used classed is {:d}'.format(len(new_ids)))
    print(new_ids)
    
    valid_ids = {v : i + 1 for i, v in enumerate(new_ids)}
    print(valid_ids)

    annotationPath = os.path.join(dataRoot, 'annotations', 'instances_%s.json' % split)
    bboxAnnosPath = os.path.join(dataRoot, 'annotations', 'coco_%s_bbox_annos.json' % split)
    with open(bboxAnnosPath, 'r') as f:
        bboxAnnos = json.load(f)
    
    coco = COCO(annotationPath)

    maskSavePath = os.path.join(dataRoot, '%s_seg60' % split)
    if not os.path.exists(maskSavePath):
        os.makedirs(maskSavePath)

    count = 0
    for imageId, bboxes in bboxAnnos.items():
        # print(count)
        if count % 1000 == 0:
            print('Processing {:d}\'s seg_60 label'.format(count))
        catIds = []
        for bbox in bboxes:
            if not catIds.__contains__(bbox[4]):
                catIds.append(bbox[4])
        
        if set(voc_ids) > set(catIds):
            continue
        
        imageName = os.path.join(dataRoot, split, str(int(imageId)).zfill(12) + '.jpg')
        image = cv2.imread(imageName)
        mask = np.zeros(shape=[image.shape[0], image.shape[1]], dtype=np.uint8)
        for Id in catIds:
            if voc_ids.__contains__(Id):
                continue
            annIds = coco.getAnnIds(imgIds=int(imageId), catIds=[Id], iscrowd=None)
            anns = coco.loadAnns(annIds)

            for i in range(len(anns)):
                maskCOCO = coco.annToMask(anns[i])
                mask[maskCOCO == 1] = valid_ids[Id]
        
        cv2.imwrite(os.path.join(maskSavePath, str(int(imageId)).zfill(12) + '.png'), mask)
        count += 1

    print('Done.')

if __name__=="__main__":
    parseCOCOBBox(dataRoot='/home/vious/data/cocoDataset/', split='train2017')
    genBboxLabel(dataRoot='/home/vious/data/cocoDataset/', split='train2017')
    parseCOCOLabel(dataRoot='/home/vious/data/cocoDataset/', split='train2017')
