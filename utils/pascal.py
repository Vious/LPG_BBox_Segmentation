import xml.etree.ElementTree as ET

def parse_xml(anno_path):
    CLASSES = ('background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    index_map = dict(zip(CLASSES, range(len(CLASSES))))

    tree = ET.parse(anno_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    # print(width,height)

    def validate_label(xmin, ymin, xmax, ymax, width, height):
            """Validate labels."""
            assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
            assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
            assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
            assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)
        
    label = []
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text)
        cls_name = obj.find('name').text.strip().lower()
        if cls_name not in CLASSES:
            continue
        cls_id = index_map[cls_name]
        
        xml_box = obj.find('bndbox')
        xmin = (int(xml_box.find('xmin').text) - 1)
        ymin = (int(xml_box.find('ymin').text) - 1)
        xmax = (int(xml_box.find('xmax').text) - 1)
        ymax = (int(xml_box.find('ymax').text) - 1)

        try:
            validate_label(xmin, ymin, xmax, ymax, width, height)
        except AssertionError as e:
            raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
        # label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        label.append([xmin, ymin, xmax, ymax, cls_id])

    return label
