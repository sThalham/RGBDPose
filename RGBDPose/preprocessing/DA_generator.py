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
from ..utils.image import read_image_bgr, read_image_dep, preprocess_image
from collections import defaultdict

import keras
import os
import json
import numpy as np
import itertools
import cv2
import math

from ..utils.anchors import relative_rotations_targets


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class DAGenerator(Generator):
    """ Generate data from the LineMOD dataset.
    """

    def __init__(self, data_dir, set_name, real_name, crop_size, **kwargs):

        self.data_dir  = data_dir
        self.set_name  = set_name
        self.path      = os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json')
        with open(self.path, 'r') as js:
            data = json.load(js)

        #self.image_ann = []
        #for anno in data["images"]:
        #    if int(anno['id']) > 18274:
        #        self.image_ann.append(anno)
        self.image_ann = data["images"]
        anno_ann = data["annotations"]
        cat_ann = data["categories"]
        self.cats = {}
        self.image_ids = []
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        for cat in cat_ann:
            self.cats[cat['id']] = cat
        for img in self.image_ann:
            self.image_ids.append(img['id'])  # to correlate indexing to self.image_ann
        for ann in anno_ann:
            self.imgToAnns[ann['image_id']].append(ann)
            self.catToImgs[ann['category_id']].append(ann['image_id'])

        self.load_classes()

        self.path_syn = os.path.join(data_dir, 'images', 'train')
        self.path_real = os.path.join(data_dir, 'images', real_name)
        self.crop_size = crop_size

        self.image_ids_syn = os.listdir(os.path.join(self.path_syn))
        self.image_ids_syn = [img[:-8] for img in self.image_ids_syn]
        self.image_ids_syn = np.unique(np.asarray(self.image_ids_syn))
        np.random.shuffle(self.image_ids_syn)

        self.image_ids_real = os.listdir(os.path.join(self.path_real))
        self.image_ids_real = [img[:-8] for img in self.image_ids_real]
        self.image_ids_real = np.unique(np.asarray(self.image_ids_real))
        np.random.shuffle(self.image_ids_real)

        self.relative_rotations_targets = relative_rotations_targets
        self.domain = True
        self.task = True

        super(DAGenerator, self).__init__(**kwargs)

    def load_classes(self):
        """ Loads the class to label mapping (and inverse) for COCO.
        """

        categories = self.cats
        if _isArrayLike(categories):
            categories = [categories[id] for id in categories]
        elif type(categories) == int:
            categories = [categories[categories]]
        categories.sort(key=lambda x: x['id'])

        self.classes        = {}
        self.labels         = {}
        self.labels_inverse = {}
        for c in categories:
            self.labels[len(self.classes)] = c['id']
            self.labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels_rev = {}
        for key, value in self.classes.items():
            self.labels_rev[value] = key

    def size(self):

        return len(self.image_ids)

    def num_classes(self):

        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels_rev

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def inv_label_to_label(self, label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.labels_inverse[label]

    def inv_label_to_name(self, label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.label_to_label(label))

    def label_to_inv_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):

        if _isArrayLike(image_index):
            image = (self.image_ann[id] for id in image_index)
        elif type(image_index) == int:
            image = self.image_ann[image_index]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        if self.task:
            if _isArrayLike(image_index):
                image_info = (self.image_ann[id] for id in image_index)
            elif type(image_index) == int:
                image_info = self.image_ann[image_index]
            path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
            path = path[:-4] + '_rgb' + path[-4:]
        else:
            if self.domain:
                path = os.path.join(self.path_real, self.image_ids_real[image_index])
            else:
                path = os.path.join(self.path_syn, self.image_ids_syn[image_index])
            path = path + '_rgb.jpg'

        return read_image_bgr(path)

    def load_image_dep(self, image_index):
        """ Load an image at the image_index.
        """
        if self.task:
            if _isArrayLike(image_index):
                image_info = (self.image_ann[id] for id in image_index)
            elif type(image_index) == int:
                image_info = self.image_ann[image_index]
            path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
            path = path[:-4] + '_dep.png'# + path[-4:]
            #path = path[:-4] + '_dep' + path[-4:]
        else:
            if self.domain:
                path = os.path.join(self.path_real, self.image_ids_real[image_index])
                path = path + '_dep.png'
            else:
                path = os.path.join(self.path_syn, self.image_ids_syn[image_index])
                path = path + '_dep.png'

        return read_image_dep(path)

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
            CHECK DONE HERE: Annotations + images correct
        """
        # get ground truth annotations
        self.task = False
        ids = self.image_ids[image_index]
        ids = ids if _isArrayLike(ids) else [ids]

        lists = [self.imgToAnns[imgId] for imgId in ids if imgId in self.imgToAnns]
        anns = list(itertools.chain.from_iterable(lists))

        annotations     = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4)), 'poses': np.empty((0, 6)), 'segmentations': np.empty((0, 16)), 'rotations': np.empty((0, 4))}

        for idx, a in enumerate(anns):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate([annotations['labels'], [self.inv_label_to_label(a['category_id'])]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)
            annotations['poses'] = np.concatenate([annotations['poses'], [[
                a['pose'][0],
                a['pose'][1],
                a['pose'][2],
                a['pose'][3],
                a['pose'][4],
                a['pose'][5],
            ]]], axis=0)
            annotations['segmentations'] = np.concatenate([annotations['segmentations'], [[
                a['segmentation'][0],
                a['segmentation'][1],
                a['segmentation'][2],
                a['segmentation'][3],
                a['segmentation'][4],
                a['segmentation'][5],
                a['segmentation'][6],
                a['segmentation'][7],
                a['segmentation'][8],
                a['segmentation'][9],
                a['segmentation'][10],
                a['segmentation'][11],
                a['segmentation'][12],
                a['segmentation'][13],
                a['segmentation'][14],
                a['segmentation'][15],
            ]]], axis=0)
            annotations['rotations'] = np.concatenate([annotations['rotations'], [[0, 0, 0, 0]]], axis=0)

        return annotations

    def load_annotations_RR(self, image_index):
        """ Load annotations for an image_index.
            CHECK DONE HERE: Annotations + images correct
        """
        self.task = True
        # get ground truth annotations
        ids = self.image_ids[image_index]
        ids = ids if _isArrayLike(ids) else [ids]

        lists = [self.imgToAnns[imgId] for imgId in ids if imgId in self.imgToAnns]
        anns = list(itertools.chain.from_iterable(lists))

        annotations     = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4)), 'poses': np.empty((0, 6)), 'segmentations': np.empty((0, 16)), 'rotations': np.empty((0, 4))}

        return annotations
    

    def random_rotation(self, image_group, annotations_group, sub_batches):

        new_image_group = []
        new_annotations_group = []

        for index in range(len(image_group)):
            # transform a single group entry
            img_tuple = image_group[index]

            for crop_idx in range(sub_batches):

                if self.crop_size >= img_tuple[0].shape[0]:
                    off_y = 0
                else:
                    off_y = np.random.randint(0, (img_tuple[0].shape[0] - self.crop_size))

                off_x = np.random.randint(0, (img_tuple[0].shape[1] - self.crop_size))

                img_crop_0 = img_tuple[0][off_y:(off_y + self.crop_size), off_x:(off_x + self.crop_size)]
                img_crop_1 = img_tuple[1][off_y:(off_y + self.crop_size), off_x:(off_x + self.crop_size)]

                img_crop_0 = img_crop_0.astype(np.uint8)
                img_crop_1 = img_crop_1.astype(np.uint8)

                angle_0 = np.random.choice([0, math.pi*0.5, math.pi, math.pi + math.pi*0.5], 1)
                angle_1 = np.random.choice([0, math.pi*0.5, math.pi, math.pi + math.pi*0.5], 1)
                relative_angle = angle_0 - angle_1

                if angle_0 > 0:
                    if angle_0 == (math.pi*0.5):
                        img_crop_0 = cv2.rotate(img_crop_0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif angle_0 == (math.pi):
                        img_crop_0 = cv2.rotate(img_crop_0, cv2.ROTATE_180)
                    elif angle_0 == (math.pi + math.pi*0.5):
                        img_crop_0 = cv2.rotate(img_crop_0, cv2.ROTATE_90_CLOCKWISE)

                if angle_1 > 0:
                    if angle_1 == (math.pi*0.5):
                        img_crop_1 = cv2.rotate(img_crop_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif angle_1 == (math.pi):
                        img_crop_1 = cv2.rotate(img_crop_1, cv2.ROTATE_180)
                    elif angle_1 == (math.pi + math.pi*0.5):
                        img_crop_1 = cv2.rotate(img_crop_1, cv2.ROTATE_90_CLOCKWISE)

                img_crop_1 = img_crop_1 * (255.0/np.nanmax(img_crop_1))
                img_crop_1 = np.repeat(img_crop_1[:, :, np.newaxis], 3, axis=2)
                new_image_group.append([img_crop_0, img_crop_1])

                these_labels = np.zeros((4))
                these_labels[int(relative_angle/(math.pi/2))] = 1
                annotations_group[index*sub_batches+crop_idx]['rotations'] = these_labels

        return new_image_group, annotations_group

    def compute_input_output_RR(self, group):

        self.domain = not self.domain
        sub_batches = 4
        batch_size = len(group)

        # load images and annotations
        image_group       = self.load_image_group(group) # image group is now [image_rgb, image_dep]
        annotations_group = [self.load_annotations_RR(image_index) for image_index in range(len(group)*sub_batches)]

        #print('len image_group: ', len(image_group))
        #print('len annotations_group: ', len(annotations_group))

        # check validity of annotations
        #image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_rotation(image_group, annotations_group, sub_batches)

        # perform preprocessing steps
        #image_group, annotations_group = self.preprocess_group(image_group, annotations_group)
        for i in range(len(image_group)):
            image_group[i][0] = preprocess_image(image_group[i][0])
            image_group[i][1] = preprocess_image(image_group[i][1])

            #image_group[i][0], image_scale0 = self.resize_image(image_group[i][0])
            #image_group[i][1], image_scale0 = self.resize_image(image_group[i][1])

            image_group[i][0] = keras.backend.cast_to_floatx(image_group[i][0])
            image_group[i][1] = keras.backend.cast_to_floatx(image_group[i][1])

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets_RR(image_group, annotations_group)

        return inputs, targets


    def compute_targets_RR(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        max_shape = tuple(max(image[0].shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.relative_rotations_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)


    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        if self.task:
            inputs, targets = self.compute_input_output(group)
        else:
            inputs, targets = self.compute_input_output_RR(group)
        #self.task = not self.task

        return inputs, targets
