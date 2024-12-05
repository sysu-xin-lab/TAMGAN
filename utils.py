import numpy as np
import imageio
import os
from tqdm import tqdm
import cv2
import random
import gdal

### define some global variables ###
group_num = 8
group_size = 46
ori_train_size = 400
ori_test_size = 2400
bands = 4
patch_size = 64
patch_stride = 48
crop_size = (patch_size - patch_stride) // 2
num_patches_x = ori_train_size // patch_stride
num_patches_y = ori_train_size // patch_stride
num_patches = num_patches_x * num_patches_y


### read train data batchly ###
def readtraindata(datapath):
    ori_train = np.zeros([group_num, group_size, ori_train_size, ori_train_size, bands], dtype=np.float32)
    cur_num = 0
    for root, _, files in os.walk(datapath):
        files.sort()
        if len(files) != 0:
            cur_image = 0
            for file in files:
                file_name = root + '/' + file
                ori_train[cur_num, cur_image, :, :, :] = imageio.imread(file_name)[:, :, 0:4]
                cur_image += 1
            cur_num += 1
    return ori_train


### generate the data and the lebels ###
def generate_dataset(array):
    ### calculate the mean and std value ###
    mean = np.zeros([array.shape[0], group_size, bands], dtype=np.float32)
    std = np.zeros([array.shape[0], group_size, bands], dtype=np.float32)
    for i in range(array.shape[0]):
        for j in range(group_size):
            mean[i, j, :] = np.mean(array[i, j].reshape(-1, bands), axis=0)
            std[i, j, :] = np.std(array[i, j].reshape(-1, bands), axis=0)
    ### difine a new input for noise ###
    noise_image = np.zeros([array.shape[0], group_size, ori_train_size, ori_train_size, bands], dtype=np.float32)
    for i in tqdm(range(array.shape[0])):
        for j in range(group_size):
            for k in range(bands):
                noise_image[i, j, :, :, k] = np.random.normal(mean[i, j, k],
                                                              std[i, j, k], [ori_train_size, ori_train_size])
    ### define some arrays ###
    data_ori = np.zeros([array.shape[0] * num_patches, group_size, patch_size, patch_size, bands], dtype=np.float32)
    noise = np.zeros([array.shape[0] * num_patches, group_size, patch_size, patch_size, bands], dtype=np.float32)
    mask = np.zeros([array.shape[0] * num_patches, group_size, patch_size, patch_size, bands], dtype=np.float32)
    ### cut into (None,64,64,bands) ###
    for i in tqdm(range(array.shape[0])):
        for j in range(num_patches_x):
            id_x = j * patch_stride
            for k in range(num_patches_y):
                id_y = k * patch_stride
                data_ori[i * num_patches + j * num_patches_y + k] = \
                    array[i, :, id_x:id_x + patch_size, id_y:id_y + patch_size, :]
                noise[i * num_patches + j * num_patches_y + k] = \
                    noise_image[i, :, id_x:id_x + patch_size, id_y:id_y + patch_size, :]
    ### generate masks randomly and add them into patches ###
    for i in tqdm(range(data_ori.shape[0])):
        for j in range(data_ori.shape[1]):
            mask_size = random.randint(20, 25)
            mask_x = random.randint(mask_size - 5, patch_size - mask_size + 5)
            mask_y = random.randint(mask_size - 5, patch_size - mask_size + 5)
            for k in range(patch_size):
                for m in range(patch_size):
                    if mask_x - mask_size <= k <= mask_x + mask_size and mask_y - mask_size <= m <= mask_y + mask_size:
                        mask[i, j, k, m, :] = 0
                    else:
                        mask[i, j, k, m, :] = 1
    data = data_ori * mask + noise * (1 - mask)
    label = np.concatenate([data_ori, mask], axis=-1)
    return data, label

### read test data ###
def readtestdata(datapath):
    ori_test = np.zeros([1, group_size, ori_test_size, ori_test_size, bands], dtype=np.float32)
    qa = np.zeros([1, group_size, ori_test_size, ori_test_size], dtype=np.float32)
    noise = np.zeros([1, group_size, ori_test_size, ori_test_size, bands], dtype=np.float32)
    mask = np.ones([1, group_size, ori_test_size, ori_test_size, bands], dtype=np.float32)

    ### read reflectance information ###
    for _, _, files in os.walk(datapath):
        files.sort()
        cur_image = 0
        for file in tqdm(files):
            file_name = datapath + file
            #############################################################################################
            ori_test[0, cur_image, :, :, 0:4] = imageio.imread(file_name)[:, :, 0:4] / 10000
            qa[0, cur_image] = imageio.imread(file_name)[:, :, 11]
            ##############################################################################################
            cur_image += 1
    ### mask cloudy pixels ###
    #######################################################################################################
    for u in tqdm(range(group_size)):
        for i in range(ori_test_size):
            for j in range(ori_test_size):
                ########################################################################################################
                if (int(qa[0, u, i, j]) & 1 != 0 or int(qa[0, u, i, j]) & 4 != 0 or int(
                        qa[0, u, i, j]) & 512 != 0 or int(qa[0, u, i, j]) & 1024 != 0
                        or int(qa[0, u, i, j]) & 4096 != 0 or int(qa[0, u, i, j]) & 32768 != 0):
                ########################################################################################################
                    mask[0, u, i, j, :] = 0
    ### eroded masks ###
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for u in range(group_size):
        for i in range(bands):
            mask[0, u, :, :, i] = cv2.erode(mask[0, u, :, :, i], kernel)
    test_ori = ori_test * mask
    #######################################################################################################
    ## fill in random noise ###
    for u in tqdm(range(group_size)):
        for i in range(bands):
            array = test_ori[0, u, :, :, i].flatten()
            array = array[array != 0]
            test_mean = np.mean(array, axis=0)
            test_std = np.std(array, axis=0)
            noise[0, u, :, :, i] = np.random.normal(test_mean, test_std, [ori_test_size, ori_test_size])
    test_data = ori_test * mask + noise * (1 - mask)
    return test_data, ori_test, mask

### write array to tif ###
def writetif(newRasterfn, array):
    driver = gdal.GetDriverByName('GTiff')
    cols = array.shape[1]
    rows = array.shape[0]
    band_count = array.shape[2]
    outRaster = driver.Create(newRasterfn, cols, rows, band_count, gdal.GDT_Float32)
    total = band_count + 1
    for index in range(1, total):
        data = array[:, :, index - 1]
        out_band = outRaster.GetRasterBand(index)
        out_band.WriteArray(data)
        out_band.FlushCache()
        out_band.ComputeBandStats(False)
    del outRaster
