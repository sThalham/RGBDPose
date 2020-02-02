#!/usr/bin/env python

import sys
import os
import subprocess
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage
import math
import datetime
import copy
import transforms3d as tf3d
import time
from pathlib import Path
import geometry
from shutil import copyfile

depSca = 1.0
resX = 720
resY = 540
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899
depthCut = 2000.0


def draw_axis(img, cam_R, cam_T):
    # unit is mm
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)

    rotMat = tf3d.quaternions.quat2mat(cam_R)
    rot, _ = cv2.Rodrigues(rotMat)

    tra = cam_T

    K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3,3)

    axisPoints, _ = cv2.projectPoints(points, rot, tra, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img


def toPix(translation):

    xpix = ((translation[0] * fxkin) / translation[2]) + cxkin
    ypix = ((translation[1] * fykin) / translation[2]) + cykin
    #zpix = translation[2] * 0.001 * fxkin

    return [xpix, ypix]


def encodeImage(depth):
    img = np.zeros((resY, resX, 3), dtype=np.uint8)

    normImg, depImg = get_normal(depth, fxkin, fykin, cxkin, cykin, for_vis=True)
    img[:, :, 0] = compute_disparity(depImg)
    img[:, :, 1] = encode_area(depImg)
    img[:, :, 2] = compute_angle2gravity(normImg, depImg)

    return img


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

    return cloud_final


def encode_area(depth):

    onethird = cv2.resize(depth, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
    pc = create_point_cloud(onethird, fxkin, fykin, cxkin, cykin, 1.0)
    cloud = pcl.PointCloud(np.array(pc, dtype=np.float32))
    CPPInput = "/home/sthalham/workspace/proto/python_scripts/CPP_workaround/bin/tempCloud.pcd"
    CPPOutput = "/home/sthalham/workspace/proto/python_scripts/CPP_workaround/bin/output.pcd"
    pcl.save(cloud, CPPInput)

    args = ("/home/sthalham/workspace/proto/python_scripts/CPP_workaround/bin/conditional_euclidean_clustering", CPPInput, CPPOutput)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    cloudNew = pcl.load_XYZI(CPPOutput)
    pcColor = cloudNew.to_array()
    inten = pcColor[:, 3]
    inten = np.reshape(inten, (int(resY/3), int(resX/3)))

    clusters, surf = np.unique(inten, return_counts=True)

    flat = surf.flatten()
    flat.sort()
    area_ref = np.mean(flat[0:-1])
    area_max = np.nanmax(surf)
    areaCol = np.ones(inten.shape, dtype=np.uint8)

    for i, cl in enumerate(clusters):
        if surf[i] < area_ref:
            mask = np.where(inten == cl, True, False)
            val = 255.0 - ((surf[i] / area_ref) * 127.5)  # prob.: 255 - ...
            val = val.astype(dtype=np.uint8)
            areaCol = np.where(mask, val, areaCol)

        else:
            mask = np.where(inten == cl, True, False)
            val = 127.5 - ((surf[i] / area_max) * 126.5)  # prob.: 255 - ...
            val = val.astype(dtype=np.uint8)
            areaCol = np.where(mask, val, areaCol)

    areaCol = cv2.resize(areaCol, (resX, resY), interpolation=cv2.INTER_NEAREST)

    areaCol = np.where(depth > depthCut, 0, areaCol)

    return areaCol


def compute_disparity(depth):
    # calculate disparity
    depthFloor = 100.0
    depthCeil = depthCut

    disparity = np.ones((depth.shape), dtype=np.float32)
    disparity = np.divide(disparity, depth)
    disparity = disparity - (1 / depthCeil)
    denom = (1 / depthFloor) - (1 / depthCeil)
    disparity = np.divide(disparity, denom)
    disparity = np.where(np.isinf(disparity), 0.0, disparity)
    dispSca = disparity - np.nanmin(disparity)
    maxV = 255.0 / np.nanmax(dispSca)
    scatemp = np.multiply(dispSca, maxV)
    disp_final = scatemp.astype(np.uint8)

    return disp_final


def compute_angle2gravity(normals, depth):
    r, c, p = normals.shape
    mask = depth < depthCut
    normals[:, :, 0] = np.where(mask, normals[:, :, 0], np.NaN)
    normals[:, :, 1] = np.where(mask, normals[:, :, 1], np.NaN)
    normals[:, :, 2] = np.where(mask, normals[:, :, 2], np.NaN)

    angEst = np.zeros(normals.shape, dtype=np.float32)
    angEst[:, :, 2] = 1.0
    ang = (45.0, 45.0, 45.0, 45.0, 45.0, 15.0, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0)
    for th in ang:
        angtemp = np.einsum('ijk,ijk->ij', normals, angEst)
        angEstNorm = np.linalg.norm(angEst, axis=2)
        normalsNorm = np.linalg.norm(normals, axis=2)
        normalize = np.multiply(normalsNorm, angEstNorm)
        angDif = np.divide(angtemp, normalize)

        np.where(angDif < 0.0, angDif + 1.0, angDif)
        angDif = np.arccos(angDif)
        angDif = np.multiply(angDif, (180 / math.pi))

        cond1 = (angDif < th)
        cond1_ = (angDif > (180.0 - th))
        cond2 = (angDif > (90.0 - th)) & (angDif < (90.0 + th))
        cond1 = np.repeat(cond1[:, :, np.newaxis], 3, axis=2)
        cond1_ = np.repeat(cond1_[:, :, np.newaxis], 3, axis=2)
        cond2 = np.repeat(cond2[:, :, np.newaxis], 3, axis=2)

        NyPar1 = np.extract(cond1, normals)
        NyPar2 = np.extract(cond1_, normals)
        NyPar = np.concatenate((NyPar1, NyPar2))
        npdim = (NyPar.shape[0] / 3)
        NyPar = np.reshape(NyPar, (int(npdim), 3))
        NyOrt = np.extract(cond2, normals)
        nodim = (NyOrt.shape[0] / 3)
        NyOrt = np.reshape(NyOrt, (int(nodim), 3))

        cov = (np.transpose(NyOrt)).dot(NyOrt) - (np.transpose(NyPar)).dot(NyPar)
        u, s, vh = np.linalg.svd(cov)
        angEst = np.tile(u[:, 2], r * c).reshape((r, c, 3))

    angDifSca = angDif - np.nanmin(angDif)
    maxV = 255.0 / np.nanmax(angDifSca)
    scatemp = np.multiply(angDifSca, maxV)
    gImg = scatemp.astype(np.uint8)
    gImg[gImg is np.NaN] = 0

    return gImg


def get_normal(depth_refine, fx=-1, fy=-1, cx=-1, cy=-1, for_vis=True):
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]

    # inpainting
    scaleOri = np.amax(depth_refine)
    print(scaleOri)

    inPaiMa = np.where(depth_refine == 0.0, 255, 0)
    inPaiMa = inPaiMa.astype(np.uint8)
    inPaiDia = 5.0
    depth_refine = depth_refine.astype(np.float32)
    depPaint = cv2.inpaint(depth_refine, inPaiMa, inPaiDia, cv2.INPAINT_NS)

    depNorm = depPaint - np.amin(depPaint)
    rangeD = np.amax(depNorm)
    depNorm = np.divide(depNorm, rangeD)
    depth_refine = np.multiply(depNorm, scaleOri)

    depth_inp = copy.deepcopy(depth_refine)

    centerX = cx
    centerY = cy

    constant = 1 / fx
    uv_table = np.zeros((res_y, res_x, 2), dtype=np.int16)
    column = np.arange(0, res_y)

    uv_table[:, :, 1] = np.arange(0, res_x) - centerX  # x-c_x (u)
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY  # y-c_y (v)
    uv_table_sign = np.copy(uv_table)
    uv_table = np.abs(uv_table)

    # kernel = np.ones((5, 5), np.uint8)
    # depth_refine = cv2.dilate(depth_refine, kernel, iterations=1)
    # depth_refine = cv2.medianBlur(depth_refine, 5 )
    depth_refine = ndimage.gaussian_filter(depth_refine, 2)  # sigma=3)
    # depth_refine = ndimage.uniform_filter(depth_refine, size=11)

    # very_blurred = ndimage.gaussian_filter(face, sigma=5)
    v_x = np.zeros((res_y, res_x, 3))
    v_y = np.zeros((res_y, res_x, 3))
    normals = np.zeros((res_y, res_x, 3))

    dig = np.gradient(depth_refine, 2, edge_order=2)
    v_y[:, :, 0] = uv_table_sign[:, :, 1] * constant * dig[0]
    v_y[:, :, 1] = depth_refine * constant + (uv_table_sign[:, :, 0] * constant) * dig[0]
    v_y[:, :, 2] = dig[0]

    v_x[:, :, 0] = depth_refine * constant + uv_table_sign[:, :, 1] * constant * dig[1]
    v_x[:, :, 1] = uv_table_sign[:, :, 0] * constant * dig[1]
    v_x[:, :, 2] = dig[1]

    cross = np.cross(v_x.reshape(-1, 3), v_y.reshape(-1, 3))
    norm = np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)
    # norm[norm == 0] = 1

    cross = cross / norm
    cross = cross.reshape(res_y, res_x, 3)
    cross = np.abs(cross)
    cross = np.nan_to_num(cross)

    #cross[depth_refine <= 200] = 0  # 0 and near range cut
    cross[depth_refine > depthCut] = 0  # far range cut
    if not for_vis:
        scaDep = 1.0 / np.nanmax(depth_refine)
        depth_refine = np.multiply(depth_refine, scaDep)
        cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_refine - 0.5))  # nearer has higher intensity
        cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_refine - 0.5))
        cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_refine - 0.5))
        scaCro = 255.0 / np.nanmax(cross)
        cross = np.multiply(cross, scaCro)
        cross = cross.astype(np.uint8)

    return cross, depth_refine, depth_inp


def create_BB(rgb):

    imgray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mask = imgray > 25

    oneA = np.ones(imgray.shape)
    masked = np.where(mask, oneA, 0)

    kernel = np.ones((9, 9), np.uint8)
    mask_dil = cv2.dilate(masked, kernel, iterations=1)

    im2, contours, hier = cv2.findContours(np.uint8(mask_dil), 1, 2)

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    area = cv2.contourArea(box)

    # cv2.drawContours(rgb, [box], -1, (170, 160, 0), 2)
    # cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    bb = [int(x),int(y),int(w),int(h)]

    return cnt, bb, area, mask_dil


if __name__ == "__main__":

    dataset = 'linemod'
    root = "/home/sthalham/data/LINEMOD/test/"  # path to train samples, depth + rgb
    target = '/home/sthalham/data/prepro/linemod_RGBD_V3_DA/'
    samples = 12655
    visu = False

    sub = os.listdir(root)

    now = datetime.datetime.now()
    dateT = str(now)

    times = []
    gloCo = 0

    for s in sub:

        rgbPath = root + s + "/rgb/"
        depPath = root + s + "/depth/"
        gtPath = root + s + "/gt.yml"
        infoPath = root + s + "/info.yml"

        with open(infoPath, 'r') as stream:
            opYML = yaml.load(stream)

        with open(gtPath, 'r') as streamGT:
            gtYML = yaml.load(streamGT)

        subsub = os.listdir(rgbPath)[:samples]

        counter = 0
        for ss in subsub:

            start_time = time.time()
            gloCo = gloCo + 1

            imgname = ss
            rgbImgPath = rgbPath + ss
            depImgPath = depPath + ss
            #print(rgbImgPath)

            if ss.startswith('000'):
                ss = ss[3:]
            elif ss.startswith('00'):
                ss = ss[2:]
            elif ss.startswith('0'):
                ss = ss[1:]
            ss = ss[:-4]


            calib = opYML[int(ss)]
            K = calib["cam_K"]
            depSca = calib["depth_scale"]
            fxca = K[0]
            fyca = K[4]
            cxca = K[2]
            cyca = K[5]
            #cam_R = calib["cam_R_w2c"]
            #cam_T = calib["cam_t_w2c"]

            #########################
            # Prepare the stuff
            #########################

            # read images and create mask
            rgbImg = cv2.imread(rgbImgPath)
            depImg = cv2.imread(depImgPath, cv2.IMREAD_UNCHANGED)
            #depImg = cv2.resize(depImg, None, fx=1 / 2, fy=1 / 2)
            rows, cols = depImg.shape
            depImg = np.multiply(depImg, depSca)
            #print(np.amax(depImg))

            # create image number and name
            template = '00000'
            s = int(s)
            ssm = int(ss) + 1
            pre = (s-1) * 1296
            img_id = pre + ssm
            tempSS = template[:-len(str(img_id))]

            imgNum = str(img_id)
            imgNam = tempSS + imgNum + '.jpg'
            iname = str(imgNam)

            gtImg = gtYML[int(ss)]

            fileName = target + 'images/lm_real/' + imgNam[:-4] + '_dep.jpg'

            myFile = Path(fileName)
            if myFile.exists():
                print('File exists, skip encoding, ', fileName)
            else:

                    scaCro = 255/np.nanmax(depImg)
                    cross = np.multiply(depImg, scaCro)
                    dep_sca = cross.astype(np.uint8)
                    imgI = np.repeat(dep_sca[:, :, np.newaxis], 3, 2)

                    rgb_name = fileName[:-8] + '_rgb.jpg'
                    cv2.imwrite(rgb_name, rgbImg)
                    cv2.imwrite(fileName, imgI)
                    print("storing image in : ", fileName)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            meantime = sum(times) / len(times)
            eta = ((samples - gloCo) * meantime) / 60
            if gloCo % 100 == 0:
                print('eta: ', eta, ' min')
                times = []

            if visu is True:
                img = cv2.imread(rgbImgPath)
                #img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
                for i, bb in enumerate(bbox_vis):
                    cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0]) + int(bb[2]), int(bb[1]) + int(bb[3])),
                                  (255, 255, 255), 3)
                    cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0]) + int(bb[2]), int(bb[1]) + int(bb[3])),
                                  (0, 0, 0), 1)
                    #
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (int(bb[2]), int(bb[1]))
                    fontScale = 1
                    fontColor = (0, 0, 0)
                    fontthickness = 1
                    lineType = 2
                    gtText = str(cat_vis[i])

                    fontColor2 = (255, 255, 255)
                    fontthickness2 = 3
                    cv2.putText(img, gtText,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor2,
                                fontthickness2,
                                lineType)

                    cv2.putText(img, gtText,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                fontthickness,
                                lineType)

                    camR = camR_vis[i]
                    camT = camT_vis[i]
                    #t_lie = [[0.0, camR[0], camR[1]],
                    #         [-camR[0], 0.0, camR[2]],
                    #         [-camR[1], -camR[2], 0.0]]
                    #t_lie = np.asarray(t_lie, dtype=np.float32)
                    #t_eul = geometry.rotations.map_hat(t_lie)
                    t_rot = tf3d.euler.euler2quat(camR[0], camR[1], camR[2])
                    t_rot = np.asarray(t_rot, dtype=np.float32)
                    draw_axis(img, cam_R=camR, cam_T=camT)

                cv2.imwrite('/home/sthalham/visTests/testBB.jpg', img)

                print('STOP')

    print('everythings done')


