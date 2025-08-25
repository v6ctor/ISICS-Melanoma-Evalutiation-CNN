# adv. hair augmentation technique credit: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159176#961899

import numpy as np
import random
import cv2
import os

from albumentations.core.transforms_interface import ImageOnlyTransform

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        augmented = self.transform(image=image)
        return augmented["image"]

class AdvancedHairAugmentation(ImageOnlyTransform):
    def __init__(self, hairs: int = 4, hairs_folder: str = "./hairs", always_apply=False, p=0.5):
        super(AdvancedHairAugmentation, self).__init__(always_apply=always_apply, p=p)
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def apply(self, image, **params):
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return image

        height, width, _ = image.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.cvtColor(hair, cv2.COLOR_BGR2RGB)
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, image.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, image.shape[1] - hair.shape[1])
            roi = image[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            image[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
        return image

    def get_params_dependent_on_targets(self, params):
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ()
