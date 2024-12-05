import numpy as np
import model as md
import tensorflow as tf
import utils
from tqdm import tqdm
import os
import imageio
import time
start_time = time.time()
if __name__ == "__main__":
    ### settings ###
    model_path = "./model/model_g.h5"
    result_path = "./data/result/"
    mask_output_path = "./data/mask_output/"
    group_size = 46
    bands = 4
    patch_size = 64
    image_size = 2400
    edge = 12
    patch_stride = 40
    crop_size = (patch_size - patch_stride) // 2
    num_patches_x = image_size // patch_stride
    num_patches_y = image_size // patch_stride
    num_patches = num_patches_x * num_patches_y
    os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BlUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    test_path = "./data/test/ori_tif/"

    ### read test data ###
    print("read test data...")
    test_data = np.zeros([1, group_size, image_size + edge * 2, image_size + edge * 2, bands], dtype=np.float32)
    test_data[:, :, edge:image_size + edge, edge:image_size + edge, :], ori_data, mask = utils.readtestdata(test_path)

    ### extend edges ###
    for i in range(edge):
        test_data[:, :, i, edge:image_size + edge, :] = test_data[:, :, 0, edge:image_size + edge, :]
        test_data[:, :, image_size + edge * 2 - 1 - i, edge:image_size + edge, :] = test_data[:, :, image_size - 1,
                                                                                    edge:image_size + edge, :]
        test_data[:, :, edge:image_size + edge, i, :] = test_data[:, :, edge:image_size + edge, 0, :]
        test_data[:, :, edge:image_size + edge, image_size + edge * 2 - 1 - i, :] = test_data[:, :,
                                                                                    edge:image_size + edge,
                                                                                    image_size - 1, :]

    ### reconstruction ###
    print("reconstruction...")
    result = np.zeros([1, group_size, image_size + edge * 2, image_size + edge * 2, bands], dtype=np.float32)
    result_edge = np.zeros([1, group_size, image_size + edge * 2, image_size + edge * 2, bands], dtype=np.float32)
    result_weight = np.ones([1, group_size, patch_size, patch_size, bands], dtype=np.float32)
    for i in range(patch_size):
        for j in range(patch_size):
            if i <= edge * 2:
                result_weight[0, :, i, j] *= i / (edge * 2)
            elif i >= patch_size - edge * 2:
                result_weight[0, :, i, j] *= (patch_size - i) / (edge * 2)
            if j <= edge * 2:
                result_weight[0, :, i, j] *= j / (edge * 2)
            elif j >= patch_size - edge * 2:
                result_weight[0, :, i, j] *= (patch_size - j) / (edge * 2)
    model = tf.keras.models.load_model(model_path, custom_objects={"loss_fun": md.loss_fun})
    model.summary()
    for i in tqdm(range(num_patches_x)):
        id_x = patch_stride * i
        for j in range(num_patches_y):
            id_y = patch_stride * j
            result_patch = model(test_data[:, :, id_x:id_x + patch_size, id_y:id_y + patch_size, :])
            result[:, :, id_x:id_x + patch_size, id_y:id_y + patch_size] += result_patch * result_weight
            result_edge[:, :, id_x + crop_size:id_x + crop_size + patch_stride,
            id_y + crop_size:id_y + crop_size + patch_stride] = result_patch[:, :,
                                                                crop_size:-crop_size, crop_size:-crop_size, :]
    result[0, :, edge:edge*2] = result_edge[0, :, edge:edge*2]
    result[0, :, image_size:image_size+edge] = result_edge[0, :, image_size:image_size+edge]
    result[0, :, :, edge:edge*2] = result_edge[0, :, :, edge:edge*2]
    result[0, :, :, image_size:image_size+edge] = result_edge[0, :, :, image_size:image_size+edge]
    result_final = result[0, :, edge:image_size+edge, edge:image_size+edge] * (1 - mask[0]) + ori_data[0] * mask[0]

    ### save results as tif ###
    i = 0
    for _, _, files in os.walk(test_path):
        files.sort()
        for file in files:
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            utils.writetif(result_path + file, result_final[i])
            if not os.path.exists(mask_output_path):
                os.makedirs(mask_output_path)
            utils.writetif(mask_output_path + file, mask[0,i])
            i += 1
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
