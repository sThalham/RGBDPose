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
from ..utils.image import read_image_bgr, preprocess_image
from collections import defaultdict

import keras
import os
import numpy as np
import cv2
import math


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class LmRotationGenerator(Generator):
    """ Generate data from the LineMOD dataset.
    """

    def __init__(self, data_dir, set_name, crop_size, **kwargs):

        self.data_dir  = data_dir
        self.set_name  = set_name
        self.path_syn = os.path.join(data_dir, 'images', 'train')
        self.path_real = os.path.join(data_dir, 'images', set_name)
        self.crop_size = crop_size

        self.image_ids_syn = os.listdir(os.path.join(self.path_syn))
        self.image_ids_syn = [img[:-8] for img in self.image_ids_syn]
        self.image_ids_syn = np.unique(np.asarray(self.image_ids_syn))
        np.random.shuffle(self.image_ids_syn)
        print(len(self.image_ids_syn))

        self.image_ids_real = os.listdir(os.path.join(self.path_real))
        self.image_ids_real = [img[:-8] for img in self.image_ids_real]
        self.image_ids_real = np.unique(np.asarray(self.image_ids_real))
        np.random.shuffle(self.image_ids_real)
        print(len(self.image_ids_syn))

        self.domain = True

        super(LmRotationGenerator, self).__init__(**kwargs)

    def size(self):

        return len(self.image_ids_syn)

    def image_aspect_ratio(self, image_index):

        #image = self.image_ann[image_index]
        return float(640) / float(480)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        if self.domain:
            path = os.path.join(self.path_real, self.image_ids_real[image_index])
        else:
            path = os.path.join(self.path_syn, self.image_ids_syn[image_index])
        path = path + '_rgb.jpg'

        return read_image_bgr(path)

    def load_image_dep(self, image_index):
        """ Load an image at the image_index.
        """
        if self.domain:
            path = os.path.join(self.path_real, self.image_ids_real[image_index])
            path = path + '_dep.png'
        else:
            path = os.path.join(self.path_syn, self.image_ids_syn[image_index])
            path = path + '_dep.png'

        return read_image_bgr(path)

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
            CHECK DONE HERE: Annotations + images correct
        """

        annotations     = {'labels': np.zeros((4)), 'bboxes': np.empty((0, 4)), 'poses': np.empty((0, 4)), 'segmentations': np.empty((0, 4))}

        return annotations

    def random_rotation(self, image_group, annotations_group):

        for index in range(len(image_group)):
            # transform a single group entry
            img_tuple = image_group[index]

            if self.crop_size >= img_tuple[0].shape[0]:
                off_y = 0
            else:
                off_y = np.random.randint(0, (img_tuple[0].shape[0] - self.crop_size))

            off_x = np.random.randint(0, (img_tuple[0].shape[1] - self.crop_size))

            img_tuple[0] = img_tuple[0][off_y:(off_y + self.crop_size), off_x:(off_x + self.crop_size)]
            img_tuple[1] = img_tuple[1][off_y:(off_y + self.crop_size), off_x:(off_x + self.crop_size)]

            img_tuple[0] = img_tuple[0].astype(np.uint8)
            img_tuple[1] = img_tuple[1].astype(np.uint8)

            angle_0 = np.random.choice([0, math.pi*0.5, math.pi, math.pi + math.pi*0.5], 1)
            angle_1 = np.random.choice([0, math.pi*0.5, math.pi, math.pi + math.pi*0.5], 1)
            relative_angle = angle_0 - angle_1

            if angle_0 > 0:
                if angle_0 == (math.pi*0.5):
                    img_tuple[0] = cv2.rotate(img_tuple[0], cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle_0 == (math.pi):
                    img_tuple[0] = cv2.rotate(img_tuple[0], cv2.ROTATE_180)
                elif angle_0 == (math.pi + math.pi*0.5):
                    img_tuple[0] = cv2.rotate(img_tuple[0], cv2.ROTATE_90_CLOCKWISE)

            if angle_1 > 0:
                if angle_1 == (math.pi*0.5):
                    img_tuple[1] = cv2.rotate(img_tuple[1], cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle_1 == (math.pi):
                    img_tuple[1] = cv2.rotate(img_tuple[1], cv2.ROTATE_180)
                elif angle_1 == (math.pi + math.pi*0.5):
                    img_tuple[1] = cv2.rotate(img_tuple[1], cv2.ROTATE_90_CLOCKWISE)

            img_tuple[1] = img_tuple[1] * (255.0/np.nanmax(img_tuple[1]))
            image_group[index] = [img_tuple[0], img_tuple[1]]

            these_labels = np.zeros((4))
            these_labels[int(relative_angle/(math.pi/2))] = 1
            annotations_group[index]['labels'] = these_labels

        return image_group, annotations_group

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image[0].shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            13
        )

        return list(batches)

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """

        self.domain = not self.domain

        # load images and annotations
        image_group       = self.load_image_group(group) # image group is now [image_rgb, image_dep]
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        #image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_rotation(image_group, annotations_group)

        # perform preprocessing steps
        #image_group, annotations_group = self.preprocess_group(image_group, annotations_group)
        for i in range(len(image_group)):
            image_group[i][0] = preprocess_image(image_group[i][0])
            image_group[i][1] = preprocess_image(image_group[i][1])

            image_group[i][0] = keras.backend.cast_to_floatx(image_group[i][0])
            image_group[i][1] = keras.backend.cast_to_floatx(image_group[i][1])

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets
