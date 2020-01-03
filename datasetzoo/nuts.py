from pathlib import Path
import os
import cv2
import numpy as np
import glob
from adaptis.data.base import BaseDataset


class ClothesSegDataset(BaseDataset):
    def __init__(self, dataset_rootpath, split='train', **kwargs):
        super(ClothesSegDataset, self).__init__(**kwargs)

        self.dataset_rootpath = Path(dataset_rootpath)
        self.dataset_split = split
        
        #Toy Dataset
        # self.dataset_samples = []
        # images_path = sorted((self.dataset_path / split).rglob('*rgb.png'))
        # for image_path in images_path:
        #     image_path = str(image_path)
        #     mask_path = image_path.replace('rgb.png', 'im.png')
        #     self.dataset_samples.append((image_path, mask_path))
        
        # Clothes Dataset
        self.dataset_samples = []
        images_path = glob.glob(os.path.join(self.dataset_rootpath,"Images","*.jpg"))
        for image_path in images_path:
            mask_name = image_path.split("/")[-1].replace('jpg', 'png')
            mask_path = os.path.join(self.dataset_rootpath,"Annotations",mask_name)
            self.dataset_samples.append((image_path, mask_path))
        self.shape = (185,185)

    def get_sample(self, index):
        image_path, mask_path = self.dataset_samples[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # shape_ = np.array(image.shape[:2])#np.array([image.shape[0],image.shape[1]])
        # scale_ = 1.0*min(shape_)/self.short_side
        # shape_ = tuple([int(i/scale_) for i in shape_])
        image = cv2.resize(image,self.shape)
        instances_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        instances_mask = cv2.resize(instances_mask, self.shape)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)
        
        sample = {'image': image}
        if self.with_segmentation:
            semantic_segmentation = (instances_mask > 0).astype(np.int32)
            sample['semantic_segmentation'] = semantic_segmentation
        else:
            instances_mask += 1

        instances_ids = self.get_unique_labels(instances_mask, exclude_zero=True)
        instances_info = {
            x: {'class_id': 1, 'ignore': False}
            for x in instances_ids
        }

        sample.update({
            'instances_mask': instances_mask,
            'instances_info': instances_info,
        })

        return sample

    @property
    def stuff_labels(self):
        return [0]

    @property
    def things_labels(self):
        return [1]
