import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_ROOT = '../data/VOCdevkit/'


class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        # 逐个遍历每个object
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            # 将边界框信息转换为0到1之间
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            # 获取类别标签-编号
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res


class VOCDetection(data.Dataset):
    def __init__(self, root, img_size=None,
                 image_sets=None,
                 transform=None,
                 target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        if image_sets is None:
            image_sets = [('2012', 'trainval'),
                          ('2007', 'trainval')]
        self.root = root
        self.img_size = img_size
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        # 动态修改路径
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, item):
        im, gt, h, w = self.pull_item(item)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        # img_id (文件路径，文件名）
        # 两个参数，正好动态修改self._annopath 与 self._imgpath
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channel = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            #  bgr to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            # 将boxes与labels水平拼接 目标数*（box的4个边+类别数label)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id

    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt