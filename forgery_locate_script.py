#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:08:45 2017
Please cite our paper as:
@inproceedings{DBLP:conf/ih/LiuGZC18,
  author    = {Yaqi Liu and
               Qingxiao Guan and
               Xianfeng Zhao and
               Yun Cao},
  title     = {Image Forgery Localization based on Multi-Scale Convolutional Neural
               Networks},
  booktitle = {Proceedings of the 6th {ACM} Workshop on Information Hiding and Multimedia
               Security, Innsbruck, Austria, June 20-22, 2018},
  pages     = {85--90},
  year      = {2018},
  timestamp = {Thu, 21 Jun 2018 08:37:36 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/ih/LiuGZC18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@author: liuyaqi
"""

import sys
sys.path.insert(0,"caffe/python");
import caffe
GPU_ID = 1 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)
import numpy as np
import math as mt

import cv2
from scipy.signal import convolve2d

"Image patch extraction function"
def patch_extract(image, patch_size, stride):
    (L1,L2,L3)=image.shape
    patch_num_L1 = int(mt.floor((L1-patch_size)/stride)+1)
    patch_num_L2 = int(mt.floor((L2-patch_size)/stride)+1)
    patches_num = patch_num_L1 * patch_num_L2
    patches = np.zeros((patches_num,patch_size,patch_size,L3),dtype=float)
    
    start_l1 = 0
    end_l1 = 0
    start_l2 = 0
    end_l2 = 0
    patches_num_real = 0
    for l1 in range(0,patch_num_L1):
        for l2 in range(0,patch_num_L2):
            start_l1 = (l1)*stride
            end_l1 = start_l1 + patch_size
            start_l2 = (l2)*stride
            end_l2 = start_l2 + patch_size
            if end_l1 <= L1 and end_l2 <= L2:
                patch = image[start_l1:end_l1,start_l2:end_l2,:]
                if patches_num_real < patches_num:
                    patches[patches_num_real,:,:,:] = patch
                    patches_num_real = patches_num_real + 1
    return patches,patch_num_L1,patch_num_L2

"The function is used to map the small feature map to the full feature map with the same size as the original image."
def map_to_full(feamap, patch_num_L1, patch_num_L2, image, patch_size, stride):
    (L1,L2,L3) = image.shape
    feamap_full = np.zeros((L1,L2,1),dtype=float)
    feamap_full_num = np.zeros((L1,L2,1),dtype=float)
    start_l1 = 0
    end_l1 = 0
    start_l2 = 0
    end_l2 = 0
    for l1 in range(0,(patch_num_L1)):
        for l2 in range(0,(patch_num_L2)):
            start_l1 = (l1)*stride
            end_l1 = start_l1 + patch_size
            start_l2 = (l2)*stride;
            end_l2 = start_l2 + patch_size
            if end_l1 <= L1  and end_l2 <= L2:
                 feamap_full[start_l1:end_l1,start_l2:end_l2,:] = feamap_full[start_l1:end_l1,start_l2:end_l2,:]+feamap[l1,l2]
                 feamap_full_num[start_l1:end_l1,start_l2:end_l2,:] = feamap_full_num[start_l1:end_l1,start_l2:end_l2,:]+1
    o_l=np.where(feamap_full_num==0)
    feamap_full[o_l] = 1.0
    feamap_full_num[o_l] = 1.0
    feamap_full=feamap_full/feamap_full_num
    if end_l1 < L1:
        for l1 in range((end_l1),L1):
            feamap_full[l1,:]=feamap_full[end_l1-1,:]
    if end_l2 < L2:
        for l2 in range((end_l2),L2):
            feamap_full[:,l2]=feamap_full[:,end_l2-1]
    return feamap_full

"Mean filtering function."
def mean_filtering(feamap,kernel_size):
    n = kernel_size
    window = np.ones((n,n))
    window/=np.sum(window)
    feamap_out=convolve2d(feamap[:,:,0],window,mode='same',boundary='symm')
    return feamap_out

"Main function"
def main_func(input_image_path):
    root_path = './'
    # the model path (deploy file and caffemodel path) of model 64 * 64
    deploy_64 = root_path + 'model_64/models_64_8_5groups_norelu_deploy.prototxt'
    caffe_model_64 = root_path + 'model_64/models_SRM_iter_20000.caffemodel'
    img_path = input_image_path
    
    # load the caffe model
    net_64 = caffe.Net(deploy_64,caffe_model_64,caffe.TEST)
    image_channel = 3
    patch_size = 64
    stride = 8
    
    mean = np.zeros((image_channel,patch_size,patch_size),dtype=float)
    mean[0,:,:]=mean[0,:,:] + 115.0/255.0
    mean[1,:,:]=mean[1,:,:] + 113.0/255.0
    mean[2,:,:]=mean[2,:,:] + 102.0/255.0
    # image preprocessing
    # the shape of the image image in the deploy file is set as (1, 3, 64, 64)
    transformer = caffe.io.Transformer({'data':net_64.blobs['data'].data.shape})
    # change the order of data from (64,64,3) to (3,64,64)
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data',mean)
    transformer.set_channel_swap('data',(2,1,0))
    
    image = caffe.io.load_image(img_path)
    
    patches,patch_num_L1,patch_num_L2 = patch_extract(image, patch_size, stride)
    feamap = np.zeros((patch_num_L1,patch_num_L2),dtype=float)
    
    for l1 in range(0,(patch_num_L1)):
        for l2 in range(0,(patch_num_L2)):
            patch_idx = l1 * patch_num_L2 + l2
            net_64.blobs['data'].data[...] = transformer.preprocess('data',patches[patch_idx,:,:,:])
            net_64.forward()
            prob = net_64.blobs['prob'].data[0].flatten()
            feamap[l1,l2] = prob[0]
            
    feamap_full = map_to_full(feamap, patch_num_L1, patch_num_L2, image, patch_size, stride)
    feamap_full = mean_filtering(feamap_full,patch_size)
    (l1,l2)=feamap_full.shape
    bifeamap_full = np.zeros((l1,l2))
    bifeamap_full[np.where(feamap_full>=0.5)]=1
    return feamap_full, bifeamap_full

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: forgery_locate input_name output_name(recommend format .bmp)'
        exit(1)
    
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    (feamap,bifeamap)=main_func(input_image_path)
    feamap*=255
    bifeamap*=255
    output_biimage_path = output_image_path[0:(len(output_image_path)-4)] + '_bi.bmp'
    cv2.imwrite(output_image_path,feamap)
    cv2.imwrite(output_biimage_path,bifeamap)
