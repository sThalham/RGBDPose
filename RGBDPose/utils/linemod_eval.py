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

#from pycocotools.cocoeval import COCOeval

import keras
import numpy as np
import json
import math
import transforms3d as tf3d
import os
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import reproj, add, adi, re, te, vsd
import yaml
import time

from matplotlib import pyplot

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


# LineMOD
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


def get_evaluation_kiru(pcd_temp_,pcd_scene_,inlier_thres,tf,final_th, model_dia):#queue
    tf_pcd =np.eye(4)
    pcd_temp_.transform(tf)

    mean_temp = np.mean(np.array(pcd_temp_.points)[:, 2])
    mean_scene = np.median(np.array(pcd_scene_.points)[:, 2])
    pcd_diff = mean_scene - mean_temp

    #open3d.draw_geometries([pcd_temp_])
    # align model with median depth of scene
    new_pcd_trans = []
    for i, point in enumerate(pcd_temp_.points):
        poi = np.asarray(point)
        poi = poi + [0.0, 0.0, pcd_diff]
        new_pcd_trans.append(poi)
    tf = np.array(tf)
    tf[2, 3] = tf[2, 3] + pcd_diff
    pcd_temp_.points = open3d.Vector3dVector(np.asarray(new_pcd_trans))
    open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=10))

    pcd_min = mean_scene - (model_dia * 2)
    pcd_max = mean_scene + (model_dia * 2)
    new_pcd_scene = []
    for i, point in enumerate(pcd_scene_.points):
        if point[2] > pcd_min or point[2] < pcd_max:
            new_pcd_scene.append(point)
    pcd_scene_.points = open3d.Vector3dVector(np.asarray(new_pcd_scene))
    #open3d.draw_geometries([pcd_scene_])
    open3d.estimate_normals(pcd_scene_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=10))

    reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                                                   open3d.registration.TransformationEstimationPointToPoint(),
                                                   open3d.registration.ICPConvergenceCriteria(max_iteration = 5)) #5?
    tf = np.matmul(reg_p2p.transformation,tf)
    tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
    pcd_temp_.transform(reg_p2p.transformation)

    open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
    #open3d.draw_geometries([pcd_scene_])
    points_unfiltered = np.asarray(pcd_temp_.points)
    last_pcd_temp = []
    for i, normal in enumerate(pcd_temp_.normals):
        if normal[2] < 0:
            last_pcd_temp.append(points_unfiltered[i, :])
    if not last_pcd_temp:
        normal_array = np.asarray(pcd_temp_.normals) * -1
        pcd_temp_.normals = open3d.Vector3dVector(normal_array)
        points_unfiltered = np.asarray(pcd_temp_.points)
        last_pcd_temp = []
        for i, normal in enumerate(pcd_temp_.normals):
            if normal[2] < 0:
                last_pcd_temp.append(points_unfiltered[i, :])
    #print(np.asarray(last_pcd_temp))
    pcd_temp_.points = open3d.Vector3dVector(np.asarray(last_pcd_temp))

    open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=30))

    hyper_tresh = inlier_thres
    for i in range(4):
        inlier_thres = reg_p2p.inlier_rmse*2
        hyper_thres = hyper_tresh * 0.75
        if inlier_thres < 1.0:
            inlier_thres = hyper_tresh * 0.75
            hyper_tresh = inlier_thres
        reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                                                       open3d.registration.TransformationEstimationPointToPlane(),
                                                       open3d.registration.ICPConvergenceCriteria(max_iteration = 1)) #5?
        tf = np.matmul(reg_p2p.transformation,tf)
        tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
        pcd_temp_.transform(reg_p2p.transformation)
    inlier_rmse = reg_p2p.inlier_rmse

    #open3d.draw_geometries([pcd_temp_, pcd_scene_])

    ##Calculate fitness with depth_inlier_th
    if(final_th>0):

        inlier_thres = final_th #depth_inlier_th*2 #reg_p2p.inlier_rmse*3
        reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_, inlier_thres, np.eye(4),
                                                       open3d.registration.TransformationEstimationPointToPlane(),
                                                       open3d.registration.ICPConvergenceCriteria(max_iteration = 1)) #5?
        tf = np.matmul(reg_p2p.transformation, tf)
        tf_pcd = np.matmul(reg_p2p.transformation, tf_pcd)
        pcd_temp_.transform(reg_p2p.transformation)

    #open3d.draw_geometries([last_pcd_temp_, pcd_scene_])

    if( np.abs(np.linalg.det(tf[:3,:3])-1)>0.001):
        tf[:3,0]=tf[:3,0]/np.linalg.norm(tf[:3,0])
        tf[:3,1]=tf[:3,1]/np.linalg.norm(tf[:3,1])
        tf[:3,2]=tf[:3,2]/np.linalg.norm(tf[:3,2])
    if( np.linalg.det(tf) < 0) :
        tf[:3,2]=-tf[:3,2]

    return tf,inlier_rmse,tf_pcd,reg_p2p.fitness


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def load_pcd(cat):
    # load meshes
    #mesh_path ="/RGBDPose/linemod_13/"
    mesh_path = "/home/stefan/data/Meshes/linemod_13/"
    #mesh_path = "/home/sthalham/data/Meshes/linemod_13/"
    ply_path = mesh_path + 'obj_' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.PointCloud()
    pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
    open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    pcd_model = open3d.read_point_cloud(ply_path)

    return pcd_model, model_vsd, model_vsd_mm


def create_point_cloud(depth, fx, fy, cx, cy, ds):

    rows, cols = depth.shape

    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)

    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cy
    xP = x.reshape(rows * cols) - cx
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fy)
    xP = np.divide(xP, fx)

    cloud_final = np.transpose(np.array((xP, yP, zP)))
    #cloud_final[cloud_final[:,2]==0] = np.NaN

    return cloud_final


def boxoverlap(a, b):
    a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])
    b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

    x1 = np.amax(np.array([a[0], b[0]]))
    y1 = np.amax(np.array([a[1], b[1]]))
    x2 = np.amin(np.array([a[2], b[2]]))
    y2 = np.amin(np.array([a[3], b[3]]))

    wid = x2-x1+1
    hei = y2-y1+1
    inter = wid * hei
    aarea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    # intersection over union overlap
    ovlap = inter / (aarea + barea - inter)
    # set invalid entries to 0 overlap
    maskwid = wid <= 0
    maskhei = hei <= 0
    np.where(ovlap, maskwid, 0)
    np.where(ovlap, maskhei, 0)

    return ovlap


def evaluate_linemod(generator, model, threshold=0.05):
    threshold = 0.5

    #mesh_info = '/RGBDPose/linemod_13/models_info.yml'
    mesh_info = '/home/stefan/data/Meshes/linemod_13/models_info.yml'
    #mesh_info = '/home/sthalham/data/Meshes/linemod_13/models_info.yml'

    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)
    model_dia = np.zeros((31), dtype=np.float32)

    for key, value in yaml.load(open(mesh_info)).items():
        fac = 0.001
        x_minus = value['min_x'] * fac
        y_minus = value['min_y'] * fac
        z_minus = value['min_z'] * fac
        x_plus = value['size_x'] * fac + x_minus
        y_plus = value['size_y'] * fac + y_minus
        z_plus = value['size_z'] * fac + z_minus
        three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                  [x_plus, y_plus, z_minus],
                                  [x_plus, y_minus, z_minus],
                                  [x_plus, y_minus, z_plus],
                                  [x_minus, y_plus, z_plus],
                                  [x_minus, y_plus, z_minus],
                                  [x_minus, y_minus, z_minus],
                                  [x_minus, y_minus, z_plus]])
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter'] * fac

    results = []
    image_ids = []
    image_indices = []
    idx = 0

    tp = np.zeros((16), dtype=np.uint32)
    fp = np.zeros((16), dtype=np.uint32)
    fn = np.zeros((16), dtype=np.uint32)

    # interlude
    tp55 = np.zeros((16), dtype=np.uint32)
    fp55 = np.zeros((16), dtype=np.uint32)
    fn55 = np.zeros((16), dtype=np.uint32)

    tp6 = np.zeros((16), dtype=np.uint32)
    fp6 = np.zeros((16), dtype=np.uint32)
    fn6 = np.zeros((16), dtype=np.uint32)

    tp65 = np.zeros((16), dtype=np.uint32)
    fp65 = np.zeros((16), dtype=np.uint32)
    fn65 = np.zeros((16), dtype=np.uint32)

    tp7 = np.zeros((16), dtype=np.uint32)
    fp7 = np.zeros((16), dtype=np.uint32)
    fn7 = np.zeros((16), dtype=np.uint32)

    tp75 = np.zeros((16), dtype=np.uint32)
    fp75 = np.zeros((16), dtype=np.uint32)
    fn75 = np.zeros((16), dtype=np.uint32)

    tp8 = np.zeros((16), dtype=np.uint32)
    fp8 = np.zeros((16), dtype=np.uint32)
    fn8 = np.zeros((16), dtype=np.uint32)

    tp85 = np.zeros((16), dtype=np.uint32)
    fp85 = np.zeros((16), dtype=np.uint32)
    fn85 = np.zeros((16), dtype=np.uint32)

    tp9 = np.zeros((16), dtype=np.uint32)
    fp9 = np.zeros((16), dtype=np.uint32)
    fn9 = np.zeros((16), dtype=np.uint32)

    tp925 = np.zeros((16), dtype=np.uint32)
    fp925 = np.zeros((16), dtype=np.uint32)
    fn925 = np.zeros((16), dtype=np.uint32)

    tp95 = np.zeros((16), dtype=np.uint32)
    fp95 = np.zeros((16), dtype=np.uint32)
    fn95 = np.zeros((16), dtype=np.uint32)

    tp975 = np.zeros((16), dtype=np.uint32)
    fp975 = np.zeros((16), dtype=np.uint32)
    fn975 = np.zeros((16), dtype=np.uint32)
    # interlude end

    tp_add = np.zeros((16), dtype=np.uint32)
    fp_add = np.zeros((16), dtype=np.uint32)
    fn_add = np.zeros((16), dtype=np.uint32)

    rotD = np.zeros((16), dtype=np.uint32)
    less5 = np.zeros((16), dtype=np.uint32)
    rep_e = np.zeros((16), dtype=np.uint32)
    rep_less5 = np.zeros((16), dtype=np.uint32)
    add_e = np.zeros((16), dtype=np.uint32)
    add_less_d = np.zeros((16), dtype=np.uint32)
    vsd_e = np.zeros((16), dtype=np.uint32)
    vsd_less_t = np.zeros((16), dtype=np.uint32)

    model_pre = []

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):
        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)

        image_raw_dep = generator.load_image_dep(index)
        image_raw_dep = np.multiply(image_raw_dep, 255.0 / np.nanmax(image_raw_dep))
        image_raw_dep = np.repeat(image_raw_dep[:, :, np.newaxis], 3, 2)
        image_dep = generator.preprocess_image(image_raw_dep)
        image_dep, scale = generator.resize_image(image_dep)

        raw_dep_path = generator.load_image_dep_raw(index)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        anno = generator.load_annotations(index)
        if len(anno['labels']) > 1:
            t_cat = 2
            obj_name = '02'
            ent = np.where(anno['labels'] == 1.0)
            t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[ent][0]
            t_tra = anno['poses'][ent][0][:3]
            t_rot = anno['poses'][ent][0][3:]

        else:
            t_cat = int(anno['labels']) + 1
            obj_name = str(t_cat)
            if len(obj_name) < 2:
                obj_name = '0' + obj_name
            t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[0]
            t_tra = anno['poses'][0][:3]
            t_rot = anno['poses'][0][3:]

        print(t_cat)

        #if t_cat == 6:
        #    continue

        if t_cat == 3 or t_cat == 7:
            print(t_cat, ' ====> skip')
            continue

        # run network
        images = []
        images.append(image)
        images.append(image_dep)
        boxes, boxes3D, scores, labels = model.predict_on_batch(np.expand_dims(image_dep, axis=0))#, np.expand_dims(image_dep, axis=0)])

        print(scores)
        '''
        print(P3.shape) # 60, 80
        print(P4.shape) # 30, 40
        print(P5.shape) # 15, 20

        square = 16
        ix = 1
        magnitude_spectrum = np.zeros((60, 80))
        s_value = 0
        for _ in range(1):
            for _ in range(256):
                #if ix <7:
                    #ax = pyplot.subplot(2, 3, ix)
                    #ax.set_xticks([])
                    #ax.set_yticks([])
                    #pyplot.imshow(P3[0, :, :, ix - 1], cmap='hot')

                #ax = pyplot.subplot(4, 2, ix+1)
                #ax.set_xticks([])
                #ax.set_yticks([])
                #sp = np.fft.fft2(P3[0, :, :, ix - 1])
                #sp[0, 0] = 0.0
                #sp[0,0] = 0.0
                #print(np.abs(sp))
                #pyplot.imshow(np.abs(sp*(255/np.nanmax(sp))), cmap='gnuplot')

                f = np.fft.fft2(P3[0, :, :, ix - 1])
                s_value += np.sum(np.abs(f[1:, 1:])) / (80*60)
                #print(f.shape)
                fshift = np.fft.fftshift(f)
                #print(fshift.shape)
                magnitude_spectrum = np.abs(fshift)
                #magnitude_spectrum += np.log(np.abs(fshift))
                #pyplot.imshow(magnitude_spectrum, cmap='gnuplot')
                #pyplot.show()

                ix += 1

                #freq = np.fft.fftfreq(P3[0, :, :, ix-1].shape[0] * P3[0, :, :, ix-1].shape[1])
                #pyplot.plot(freq, sp.real, freq, sp.imag)
                #pyplot.show()
                #pyplot.imshow(P3[0, :, :, ix-1], cmap='gray')
                #print(np.nanmin(P3[0, :, :, ix - 1]), np.nanmax(P3[0, :, :, ix - 1]))
                #print(np.fft.fft(P3[0, :, :, ix - 1]))
                #var = var + np.var(P3[0, :, :, ix - 1])


        #pyplot.show()
        print(s_value/256.0)

        pyplot.imshow(magnitude_spectrum, cmap='gnuplot', vmin=0, vmax=3000)
        pyplot.show()
        '''

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # target annotation

        if obj_name != model_pre:
            point_cloud, model_vsd, model_vsd_mm = load_pcd(obj_name)
            #point_cloud.scale(1000.0)
            model_pre = obj_name

        rotD[t_cat] += 1
        rep_e[t_cat] += 1
        add_e[t_cat] += 1
        vsd_e[t_cat] += 1
        #t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[0]
        #t_tra = anno['poses'][0][:3]
        #t_rot = anno['poses'][0][3:]
        fn[t_cat] += 1
        #interlude
        fn55[t_cat] += 1
        fn6[t_cat] += 1
        fn65[t_cat] += 1
        fn7[t_cat] += 1
        fn75[t_cat] += 1
        fn8[t_cat] += 1
        fn85[t_cat] += 1
        fn9[t_cat] += 1
        fn925[t_cat] += 1
        fn95[t_cat] += 1
        fn975[t_cat] += 1

        # end interlude
        fn_add[t_cat] += 1
        fnit = True

        # compute predicted labels and scores
        for box, box3D, score, label in zip(boxes[0], boxes3D[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                continue

            if label < 0:
                continue

            cls = generator.label_to_inv_label(label)
            if cls > 5:
                cls += 2
            elif cls > 2:
                cls += 1
            else: pass

            #cls = 1
            #control_points = box3D[(cls - 1), :]
            control_points = box3D

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_inv_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
                'pose'        : control_points.tolist()
            }

            # append detection to results
            results.append(image_result)

            if cls == t_cat:
                b1 = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                b2 = np.array([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]])
                IoU = boxoverlap(b1, b2)
                # occurences of 2 or more instances not possible in LINEMOD
                if IoU > 0.5:
                    if fnit is True:
                        # interlude
                        if IoU > 0.55:
                            tp55[t_cat] += 1
                            fn55[t_cat] -= 1
                        else:
                            fp55[t_cat] += 1
                        if IoU > 0.6:
                            tp6[t_cat] += 1
                            fn6[t_cat] -= 1
                        else:
                            fp6[t_cat] += 1
                        if IoU > 0.65:
                            tp65[t_cat] += 1
                            fn65[t_cat] -= 1
                        else:
                            fp65[t_cat] += 1
                        if IoU > 0.7:
                            tp7[t_cat] += 1
                            fn7[t_cat] -= 1
                        else:
                            fp7[t_cat] += 1
                        if IoU > 0.75:
                            tp75[t_cat] += 1
                            fn75[t_cat] -= 1
                        else:
                            fp75[t_cat] += 1
                        if IoU > 0.8:
                            tp8[t_cat] += 1
                            fn8[t_cat] -= 1
                        else:
                            fp8[t_cat] += 1
                        if IoU > 0.85:
                            tp85[t_cat] += 1
                            fn85[t_cat] -= 1
                        else:
                            fp85[t_cat] += 1
                        if IoU > 0.9:
                            tp9[t_cat] += 1
                            fn9[t_cat] -= 1
                        else:
                            fp9[t_cat] += 1
                        if IoU > 0.925:
                            tp925[t_cat] += 1
                            fn925[t_cat] -= 1
                        else:
                            fp925[t_cat] += 1
                        if IoU > 0.95:
                            tp95[t_cat] += 1
                            fn95[t_cat] -= 1
                        else:
                            fp95[t_cat] += 1
                        if IoU > 0.975:
                            tp975[t_cat] += 1
                            fn975[t_cat] -= 1
                        else:
                            fp975[t_cat] += 1

                        # interlude end

                        tp[t_cat] += 1
                        fn[t_cat] -= 1
                        fnit = False

                        obj_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32) #.reshape((8, 1, 3))
                        est_points = np.ascontiguousarray(control_points.T, dtype=np.float32).reshape((8, 1, 2))

                        K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

                        #retval, orvec, otvec = cv2.solvePnP(obj_points, est_points, K, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
                        retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                           imagePoints=est_points, cameraMatrix=K,
                                                                           distCoeffs=None, rvec=None, tvec=None,
                                                                           useExtrinsicGuess=False, iterationsCount=100,
                                                                           reprojectionError=5.0, confidence=0.99,
                                                                           flags=cv2.SOLVEPNP_ITERATIVE)

                        R_est, _ = cv2.Rodrigues(orvec)
                        t_est = otvec

                        rot = tf3d.quaternions.mat2quat(R_est)
                        #pose = np.concatenate(
                        #            (np.array(t_est[:, 0], dtype=np.float32), np.array(rot, dtype=np.float32)), axis=0)

                        t_rot = tf3d.quaternions.quat2mat(t_rot)
                        R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                        t_gt = np.array(t_tra, dtype=np.float32) * 0.001

                        rd = re(R_gt, R_est)
                        xyz = te(t_gt, t_est.T)

                        '''
                        print('--------------------- ICP refinement -------------------')
                        # print(raw_dep_path)
                        image_dep = cv2.imread(raw_dep_path, cv2.IMREAD_UNCHANGED)
                        #image_icp = np.multiply(image_dep, 0.1)
                        image_icp = image_dep
                        #print(np.nanmax(image_icp))

                        pcd_img = create_point_cloud(image_icp, fxkin, fykin, cxkin, cykin, 1.0)
                        pcd_img = pcd_img.reshape((480, 640, 3))[int(b1[1]):int(b1[3]), int(b1[0]):int(b1[2]), :]
                        pcd_img = pcd_img.reshape((pcd_img.shape[0] * pcd_img.shape[1], 3))
                        todel = []

                        for i in range(len(pcd_img[:, 2])):
                            if pcd_img[i, 2] < 300:
                                todel.append(i)
                        pcd_img = np.delete(pcd_img, todel, axis=0)
                        #print(pcd_img)

                        pcd_crop = open3d.PointCloud()
                        pcd_crop.points = open3d.Vector3dVector(pcd_img)
                       # open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(
                       #     radius=2.0, max_nn=30))

                        # pcd_crop.paint_uniform_color(np.array([0.99, 0.0, 0.00]))
                        #open3d.draw_geometries([pcd_crop])

                        guess = np.zeros((4, 4), dtype=np.float32)
                        guess[:3, :3] = R_est
                        guess[:3, 3] = t_est.T * 1000.0
                        guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T


                        pcd_model = open3d.geometry.voxel_down_sample(point_cloud, voxel_size=5.0)
                        pcd_crop = open3d.geometry.voxel_down_sample(pcd_crop, voxel_size=5.0)

                        #open3d.draw_geometries([pcd_crop, pcd_model])
                        reg_p2p, _, _, _ = get_evaluation_kiru(pcd_model, pcd_crop, 50, guess, 5,
                                                               model_dia[cls] * 1000.0)
                        R_est = reg_p2p[:3, :3]
                        t_est = reg_p2p[:3, 3] * 0.001
                    
                        '''
                        pose = est_points.reshape((16)).astype(np.int16)
                        bb = b1

                        tDbox = R_gt.dot(obj_points.T).T
                        tDbox = tDbox + np.repeat(t_gt[np.newaxis, :], 8, axis=0)
                        box3D = toPix_array(tDbox)
                        tDbox = np.reshape(box3D, (16))
                        tDbox = tDbox.astype(np.uint16)

                        colGT = (0, 128, 0)
                        colEst = (255, 0, 0)

                        cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                                      (255, 255, 255), 2)

                        image = cv2.line(image, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 3)
                        image = cv2.line(image, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 3)
                        image = cv2.line(image, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                                         3)
                        image = cv2.line(image, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                                         3)
                        image = cv2.line(image, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                                         3)
                        image = cv2.line(image, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                                         3)
                        image = cv2.line(image, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                                         3)
                        image = cv2.line(image, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                                         3)
                        image = cv2.line(image, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                         colGT,
                                         3)
                        image = cv2.line(image, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                                         colGT,
                                         3)
                        image = cv2.line(image, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                                         colGT,
                                         3)
                        image = cv2.line(image, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                         colGT,
                                         3)

                        image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 5)
                        image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 5)
                        image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 5)
                        image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 5)
                        image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 5)
                        image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 5)
                        image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 5)
                        image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 5)
                        image = cv2.line(image, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                                         5)
                        image = cv2.line(image, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                                         5)
                        image = cv2.line(image, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                                         5)
                        image = cv2.line(image, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                                         5)

                        name = '/home/stefan/RGBDPose_viz/detection_LM.jpg'
                        cv2.imwrite(name, image+100)

                        print('break')


                        if not math.isnan(rd):
                            if rd < 5.0 and xyz < 0.05:
                                less5[t_cat] += 1

                        #err_vsd = vsd(R_est, t_est * 1000.0, R_gt, t_gt * 1000.0, model_vsd_mm, image_dep, K, 0.3, 20.0)
                        #if not math.isnan(err_vsd):
                        #    if err_vsd < 0.3:
                        #        vsd_less_t[t_cat] += 1

                        err_repr = reproj(K, R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        if not math.isnan(err_repr):
                            if err_repr < 5.0:
                                rep_less5[t_cat] += 1

                        if cls == 3 or cls == 7 or cls == 10 or cls == 11:
                            err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                        else:
                            err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        print(' ')
                        print('error: ', err_add, 'threshold', model_dia[cls] * 0.1)

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls] * 0.1):
                                add_less_d[t_cat] += 1

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls] * 0.15):
                                tp_add[t_cat] += 1
                                fn_add[t_cat] -= 1

                else:
                    fp[t_cat] += 1
                    fp_add[t_cat] += 1

                    fp55[t_cat] += 1
                    fp6[t_cat] += 1
                    fp65[t_cat] += 1
                    fp7[t_cat] += 1
                    fp75[t_cat] += 1
                    fp8[t_cat] += 1
                    fp85[t_cat] += 1
                    fp9[t_cat] += 1
                    fp925[t_cat] += 1
                    fp95[t_cat] += 1
                    fp975[t_cat] += 1

                print('Stop')

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])
        image_indices.append(index)
        idx += 1

    print(len(image_ids))

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    #json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    detPre = [0.0] * 16
    detRec = [0.0] * 16
    detPre_add = [0.0] * 16
    detRec_add = [0.0] * 16
    F1_add = [0.0] * 16
    less_55 = [0.0] * 16
    less_repr_5 = [0.0] * 16
    less_add_d = [0.0] * 16
    less_vsd_t = [0.0] * 16

    np.set_printoptions(precision=2)
    print('')
    for ind in range(1, 16):
        if ind == 0:
            continue

        if tp[ind] == 0:
            detPre[ind] = 0.0
            detRec[ind] = 0.0
            detPre_add[ind] = 0.0
            detRec_add[ind] = 0.0
            less_55[ind] = 0.0
            less_repr_5[ind] = 0.0
            less_add_d[ind] = 0.0
            less_vsd_t[ind] = 0.0
        else:
            detRec[ind] = tp[ind] / (tp[ind] + fn[ind]) * 100.0
            detPre[ind] = tp[ind] / (tp[ind] + fp[ind]) * 100.0
            detRec_add[ind] = tp_add[ind] / (tp_add[ind] + fn_add[ind]) * 100.0
            detPre_add[ind] = tp_add[ind] / (tp_add[ind] + fp_add[ind]) * 100.0
            F1_add[ind] = 2 * ((detPre_add[ind] * detRec_add[ind])/(detPre_add[ind] + detRec_add[ind]))
            less_55[ind] = (less5[ind]) / (rotD[ind]) * 100.0
            less_repr_5[ind] = (rep_less5[ind]) / (rep_e[ind]) * 100.0
            less_add_d[ind] = (add_less_d[ind]) / (add_e[ind]) * 100.0
            less_vsd_t[ind] = (vsd_less_t[ind]) / (vsd_e[ind]) * 100.0

        print('cat ', ind, ' rec ', detPre[ind], ' pre ', detRec[ind], ' less5 ', less_55[ind], ' repr ',
                  less_repr_5[ind], ' add ', less_add_d[ind], ' vsd ', less_vsd_t[ind], ' F1 add 0.15d ', F1_add[ind])

    dataset_recall = sum(tp) / (sum(tp) + sum(fp)) * 100.0
    dataset_precision = sum(tp) / (sum(tp) + sum(fn)) * 100.0
    dataset_recall_add = sum(tp_add) / (sum(tp_add) + sum(fp_add)) * 100.0
    dataset_precision_add = sum(tp_add) / (sum(tp_add) + sum(fn_add)) * 100.0
    F1_add_all = 2 * ((dataset_precision_add * dataset_recall_add)/(dataset_precision_add + dataset_recall_add))
    less_55 = sum(less5) / sum(rotD) * 100.0
    less_repr_5 = sum(rep_less5) / sum(rep_e) * 100.0
    less_add_d = sum(add_less_d) / sum(add_e) * 100.0
    less_vsd_t = sum(vsd_less_t) / sum(vsd_e) * 100.0

    print('ADD: ', less_add_d)

    print('IoU 05: ', sum(tp) / (sum(tp) + sum(fp)) * 100.0, sum(tp) / (sum(tp) + sum(fn)) * 100.0)
    print('IoU 055: ', sum(tp55) / (sum(tp55) + sum(fp55)) * 100.0, sum(tp55) / (sum(tp55) + sum(fn55)) * 100.0)
    print('IoU 06: ', sum(tp6) / (sum(tp6) + sum(fp6)) * 100.0, sum(tp6) / (sum(tp6) + sum(fn6)) * 100.0)
    print('IoU 065: ', sum(tp65) / (sum(tp65) + sum(fp65)) * 100.0, sum(tp65) / (sum(tp65) + sum(fn65)) * 100.0)
    print('IoU 07: ', sum(tp7) / (sum(tp7) + sum(fp7)) * 100.0, sum(tp7) / (sum(tp7) + sum(fn7)) * 100.0)
    print('IoU 075: ', sum(tp75) / (sum(tp75) + sum(fp75)) * 100.0, sum(tp75) / (sum(tp75) + sum(fn75)) * 100.0)
    print('IoU 08: ', sum(tp8) / (sum(tp8) + sum(fp8)) * 100.0, sum(tp8) / (sum(tp8) + sum(fn8)) * 100.0)
    print('IoU 085: ', sum(tp85) / (sum(tp85) + sum(fp85)) * 100.0, sum(tp85) / (sum(tp85) + sum(fn85)) * 100.0)
    print('IoU 09: ', sum(tp9) / (sum(tp9) + sum(fp9)) * 100.0, sum(tp9) / (sum(tp9) + sum(fn9)) * 100.0)
    print('IoU 0975: ', sum(tp925) / (sum(tp925) + sum(fp925)) * 100.0, sum(tp925) / (sum(tp925) + sum(fn925)) * 100.0)
    print('IoU 095: ', sum(tp95) / (sum(tp95) + sum(fp95)) * 100.0, sum(tp95) / (sum(tp95) + sum(fn95)) * 100.0)
    print('IoU 0975: ', sum(tp975) / (sum(tp975) + sum(fp975)) * 100.0, sum(tp975) / (sum(tp975) + sum(fn975)) * 100.0)

    return dataset_recall, dataset_precision, less_55, less_vsd_t, less_repr_5, less_add_d, F1_add_all
