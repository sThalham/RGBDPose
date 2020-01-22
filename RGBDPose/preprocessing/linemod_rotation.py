"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ..preprocessing.generator import Generator
from ..utils.image import read_image_bgr
from collections import defaultdict

import os
import numpy as np
import cv2
import math


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class LmRotationGenerator(Generator):
    """ Generate data from the LineMOD dataset.
    """

    def __init__(self, data_dir, set_name, **kwargs):

        self.data_dir  = data_dir
        self.set_name  = set_name

        super(LmRotationGenerator, self).__init__(**kwargs)

    def image_aspect_ratio(self, image_index):

        if _isArrayLike(image_index):
            image = (self.image_ann[id] for id in image_index)
        elif type(image_index) == int:
            image = self.image_ann[image_index]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        if _isArrayLike(image_index):
            image_info = (self.image_ann[id] for id in image_index)
        elif type(image_index) == int:
            image_info = self.image_ann[image_index]
        path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        path = path[:-4] + '_rgb' + path[-4:]

        return read_image_bgr(path)

    def load_image_dep(self, image_index):
        """ Load an image at the image_index.
        """
        if _isArrayLike(image_index):
            image_info = (self.image_ann[id] for id in image_index)
        elif type(image_index) == int:
            image_info = self.image_ann[image_index]
        path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        path = path[:-4] + '_dep' + path[-4:]

        return read_image_bgr(path)

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
            CHECK DONE HERE: Annotations + images correct
        """

        annotations     = {'labels': np.zeros((0,4))}

        return annotations

    def random_rotation(self, image_group, annotations_group):

        side = 480

        for index in range(len(image_group)):
            # transform a single group entry
            img_tuple = image_group[index]

            off_y = random.randint(0, (img_tuple[0].shape[0]-side))
            off_x = random.randint(0, (img_tuple[0].shape[1] - side))
            img_tuple[0] = img_tuple[0][off_y:(off_y+side), off_x:(off_x+side)]
            img_tuple[1] = img_tuple[1][off_y:(off_y + side), off_x:(off_x + side)]

            angle_0 = np.random.choice([0, math.pi*0.5, math.pi, math.pi + math.pi*0.5], 1)
            angle_1 = np.random.choice([0, math.pi*0.5, math.pi, math.pi + math.pi*0.5], 1)
            relative_angle = angle_0 - angle_1

            transform_0 = np.array([
                [np.cos(angle_0), -np.sin(angle_0), 0],
                [np.sin(angle_0), np.cos(angle_0), 0],
                [0, 0, 1]
            ])

            transform_1 = np.array([
                [np.cos(angle_1), -np.sin(angle_1), 0],
                [np.sin(angle_1), np.cos(angle_1), 0],
                [0, 0, 1]
            ])

            img_tuple[0] = cv2.warpAffine(
                img_tuple[0],
                transform_0,
                dsize=(img_tuple[0].shape[1], img_tuple[0].shape[0]),
            )

            img_tuple[1] = cv2.warpAffine(
                img_tuple[1],
                transform_1,
                dsize=(img_tuple[1].shape[1], img_tuple[1].shape[0]),
            )

            image_group[index] = [img_tuple[0], img_tuple[1]]
            annotations_group[index][int(relative_angle/math.pi)] = 1

        return image_group, annotations_group

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group) # image group is now [image_rgb, image_dep]
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        #image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_rotation(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets
