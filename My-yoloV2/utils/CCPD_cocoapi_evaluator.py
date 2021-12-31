import json
import tempfile

from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from data.CCPD2019 import *
from data import *
from utils.augmentations import SSDAugmentation


class CCPD_COCOAPIEvaluator:
    def __init__(self, data_dir, img_size, device, testset=False, transform=None):
        self.testset = testset

        self.dataset = CCPDDetection(
            CCPD_ROOT,
            transform=SSDAugmentation(img_size),
            splits_root='./splits/val.txt')

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=detection_collate,
            num_workers=0)
        self.img_size = img_size
        self.transform = transform
        self.device = device

    def evaluate(self, model):
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % num_images)

        # start testing
        for index in range(num_images):  # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]' % (index, num_images))

            img, id_ = self.dataset.pull_image(index)  # load a batch
            if self.transform is not None:
                x = torch.from_numpy(self.transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(self.device)
            scale = np.array([[img.shape[1], img.shape[0],
                               img.shape[1], img.shape[0]]])

            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                outputs = model(x)
                bboxes, scores, cls_inds = outputs
                bboxes *= scale
            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.class_ids[int(cls_inds[i])]

                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i])  # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score}  # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('yolov2_2017.json', 'w'))
                cocoDt = cocoGt.loadRes('yolov2_2017.json')
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            ap50, ap50_95 = cocoEval.stats[0], cocoEval.stats[1]
            print('ap50_95 : ', ap50_95)
            print('ap50 : ', ap50)

            return ap50, ap50_95
        else:
            return 0, 0
