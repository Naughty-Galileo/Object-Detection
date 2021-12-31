import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

CLASSES = ('car_plate',)

CCPD_ROOT = "../data/CCPD2019"


class CCPDTransform(object):
    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(zip(CLASSES, range(len(CLASSES))))

    def __call__(self, target, width, height):
        res = []
        name = 'car_plate'
        bboxes = target.split('_')
        bndbox = [int(bboxes[0].split('&')[0]) / width, int(bboxes[0].split('&')[1]) / height,
                  int(bboxes[1].split('&')[0]) / width, int(bboxes[1].split('&')[1]) / height]

        label_idx = self.class_to_ind[name]
        bndbox.append(label_idx)
        res += [bndbox]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class CCPDDetection(data.Dataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=CCPDTransform(),
                 dataset_name='ccpd2019',
                 splits_root='/splits/train.txt'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.splits_root = CCPD_ROOT + splits_root
        self.ids = list()
        for line in open(self.splits_root):
            self.ids.append((CCPD_ROOT, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        img_id = img_id[0] + '/' + img_id[1]

        # target = img_id.split('/')[5].split('-')[2]
        target = img_id.split('/')[4].split('-')[2]
        img = cv2.imread(img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        img_id = self.ids[index][0] + '/' + self.ids[index][1]
        return cv2.imread(img_id, cv2.IMREAD_COLOR), img_id

    def pull_anno(self, index):
        img_id = self.ids[index]
        img_id = img_id[0] + '/' + img_id[1]

        # target = img_id.split('/')[5].split('-')[2]
        bbox = img_id.split('/')[4].split('-')[2]
        img = cv2.imread(img_id)
        height, width, channels = img.shape
        target = self.target_transform(bbox, width, height)
        return target


if __name__ == "__main__":
    def base_transform(image, size, mean):
        x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x


    class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels


    img_size = 640
    # dataset
    dataset = CCPDDetection(CCPD_ROOT,
                            BaseTransform([img_size, img_size], (0, 0, 0)))

    for i in range(1000):
        im, gt, h, w = dataset.pull_item(i)
        img = im.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
        cv2.imwrite('-1.jpg', img)
        img = cv2.imread('-1.jpg')

        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        cv2.imshow('gt', img)
        cv2.waitKey(0)
