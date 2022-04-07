## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
from imageio import imwrite
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/park/kits19/starter_code')
sys.path.append('/home/park/kits19')

from utils import load_case
# from visualize import hu_to_grayscale

# hue to grayscale 변환 함수 정의
def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512

##데이터가 저장될 디렉토리 선언
dir_data = './datasets5'
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

##디렉토리 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

frame_stack_train =0
frame_stack_val =0
frame_stack_test =0

# total_frame_number = 0
#
# for fl_index in range(0, 110):
#     print('CURRENT DATASET INDEX : %d' %fl_index)
#     ## 데이터 불러오기
#     vl, seg = load_case(fl_index)
#     total_frame_number += vl.shape[0]
#     print('TOTAL FRAME : %d' %total_frame_number)

# 원하는 데이터만큼 slicing을 진행.
for fl_index in range(0, 70): #210
    print('CURRENT DATASET INDEX : %d' %fl_index)
    ## 데이터 불러오기
    vl, seg = load_case(fl_index)

    ## nii.gz 데이터 나누기 & numpy 변환 & hue scale 변환
    # vl_part = vl.slicer[180:210]
    vl_part = vl
    vl_np = vl_part.get_data()

    # seg_part = seg.slicer[180:210]
    seg_part = seg
    seg_np = seg_part.get_data()

    # Dataset Shuffle 해주기
    index_s = np.arange(vl_np.shape[0])
    np.random.shuffle(index_s)

    vl_np = vl_np[index_s]
    seg_np = seg_np[index_s]

    ## CT 이미지 Hu스케일 Gray 스케일로 변환.
    vl_gray = hu_to_grayscale(vl_np, DEFAULT_HU_MIN, DEFAULT_HU_MAX)

    total_frame = vl_np.shape[0]

    ## train, val, test set에 필요한 frame 갯수 설정.
    nframe_train = int(total_frame*0.8)
    nframe_val = int(total_frame*0.1)
    nframe_test = int(total_frame*0.1)

    ##실제로 저장하는 코드
    id_frame = np.arange(vl_np.shape[0])

    ## traning set 저장
    offset_nframe = 0


    for i in range(nframe_train):
        vl_ims = hu_to_grayscale(vl_np[i+offset_nframe:i+offset_nframe+1], DEFAULT_HU_MIN, DEFAULT_HU_MAX)
        seg_ims = seg_np[i+offset_nframe:i+offset_nframe+1]

        input_ = vl_ims[0,:,:,0]
        label_ = seg_ims[0,:,:]

        label_[label_ >= 1] = 1
        np.save(os.path.join(dir_save_train, 'label_%05d.npy' %(i+frame_stack_train)), label_)
        np.save(os.path.join(dir_save_train, 'input_%05d.npy' %(i+frame_stack_train)), input_)
        print('Making Train Data Index : %d' %(i+frame_stack_test))
        # print(np.max(label_))

    offset_nframe = nframe_train
    frame_stack_train += nframe_train

    ## Validation set 저장
    for i in range(nframe_val):
        vl_ims = hu_to_grayscale(vl_np[i+offset_nframe:i+offset_nframe+1], DEFAULT_HU_MIN, DEFAULT_HU_MAX)
        seg_ims = seg_np[i+offset_nframe:i+offset_nframe+1]

        input_ = vl_ims[0,:,:,0]
        label_ = seg_ims[0,:,:]

        label_[label_ >= 1] = 1
        np.save(os.path.join(dir_save_val, 'label_%05d.npy' %(i+frame_stack_val)), label_)
        np.save(os.path.join(dir_save_val, 'input_%05d.npy' %(i+frame_stack_val)), input_)
        print('Making Val Data Index : %d' %(i+frame_stack_val))

    ## test set 저장
    frame_stack_val += nframe_val
    offset_nframe = nframe_train + nframe_val

    for i in range(nframe_test):
        vl_ims = hu_to_grayscale(vl_np[i+offset_nframe:i+offset_nframe+1], DEFAULT_HU_MIN, DEFAULT_HU_MAX)
        seg_ims = seg_np[i+offset_nframe:i+offset_nframe+1]

        input_ = vl_ims[0,:,:,0]
        label_ = seg_ims[0,:,:]

        label_[label_ >= 1] = 1
        np.save(os.path.join(dir_save_test, 'label_%05d.npy' %(i+frame_stack_test)), label_)
        np.save(os.path.join(dir_save_test, 'input_%05d.npy' %(i+frame_stack_test)), input_)

    frame_stack_test += nframe_test

