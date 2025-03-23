import cv2
import math
import time
import os
import matplotlib.pyplot as plt


def get_filename_without_extension(filepath):
    filename = os.path.basename(filepath)  # 获取文件名
    file_ext = filename.split('.')[-1]
    filename_without_ext = os.path.splitext(filename)[0]  # 去除后缀
    dirname = os.path.dirname(filepath) # 获取路径
    result = os.path.join(dirname, filename_without_ext) # 组合路径和去除后缀的文件名
    return result, file_ext

model_map = {
    "EDSR_x4": {
        "type": "edsr",
        "path": "EDSR_x4.pb",
        "arg": 4
    },
    "EDSR_x2": {
        "type": "edsr",
        "path": "EDSR_x2.pb",
        "arg": 2
    },
    "EDSR_x3": {
        "type": "edsr",
        "path": "EDSR_x3.pb",
        "arg": 3
    },
    "ESPCN_x4": {
        "type": "espcn",
        "path": "ESPCN_x4.pb",
        "arg": 4
    },
    "ESPCN_x2": {
        "type": "espcn",
        "path": "ESPCN_x2.pb",
        "arg": 2
    },
    "ESPCN_x3": {
        "type": "espcn",
        "path": "ESPCN_x3.pb",
        "arg": 3
    },
    "FSRCNN_x4": {
        "type": "fsrcnn",
        "path": "FSRCNN_x4.pb",
        "arg": 4
    },
    "FSRCNN-small_x4": {
        "type": "fsrcnn",
        "path": "FSRCNN-small_x4.pb",
        "arg": 4
    },
    "FSRCNN_x2": {
        "type": "fsrcnn",
        "path": "FSRCNN_x2.pb",
        "arg": 2
    },
    "LapSRN_x4": {
        "type": "lapsrn",
        "path": "LapSRN_x4.pb",
        "arg": 4
    },
    "LapSRN_x2": {
        "type": "lapsrn",
        "path": "LapSRN_x2.pb",
        "arg": 2
    },
    "LapSRN_x8": {
        "type": "lapsrn",
        "path": "LapSRN_x8.pb",
        "arg": 8
    },
}

def super_res_single_image(img, model_name, preserve_size=True):
    if model_name not in model_map:
        return img
    model = model_map[model_name]
    img_size = img.shape
    start_time = time.time()
    if 'instance' not in model:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = f"SRModels/{model['path']}"
        sr.readModel(path)
        sr.setModel(model['type'], model['arg'])
        model['instance'] = sr
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"super_res_single_image():Load model '{model_name}' took: {elapsed_time:.4f} s")

    sr = model['instance']
    result = sr.upsample(img)
    if preserve_size:
        new_size = [img_size[1], img_size[0]]
        result = cv2.resize(result, dsize=new_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"super_res_single_image():Upsampling using '{model_name}' took: {elapsed_time:.4f} s")
    return result


def Test1():
    img_path = "/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/test/images/0/0_6.jpg"
    img_path_no_ext, img_ext = get_filename_without_extension(img_path)
    img = cv2.imread(img_path)
    img_size = img.shape
    plt.imshow(img[:, :, ::-1])
    plt.show()

    result_map = {}
    for model_name, model in model_map.items():
        start_time = time.time()

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = f"SRModels/{model['path']}"
        sr.readModel(path)
        model_type = model["type"]
        sr.setModel(model_type, model['arg'])
        upsample_start_time = time.time()
        result = sr.upsample(img)
        result_map[model_name] = result

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time2 = end_time - upsample_start_time
        print(f"Task '{model_name}' took: {elapsed_time:.4f} s, inference took: {elapsed_time2:.4f} s")
        # # Resized image
        # resized = cv2.resize(img, dsize=None, fx=4, fy=4)

    model_num = len(model_map.keys())
    plt.figure(figsize=(12*10, 8*10))
    item_per_row = 4
    rows = math.ceil(model_num+1 / item_per_row)

    img_idx = 1
    # Original image
    plt.subplot(rows, item_per_row, img_idx)
    plt.imshow(img[:, :, ::-1])
    img_idx += 1


    for model_name, result in result_map.items():
        # SR upscaled
        plt.subplot(rows, item_per_row, img_idx)
        plt.imshow(result[:, :, ::-1])
        plt.title(model_name)
        save_path = f'{img_path_no_ext}_{model_name}.{img_ext}'
        new_size = [img_size[1], img_size[0]]
        resized = cv2.resize(img, dsize=new_size)
        cv2.imwrite(save_path, resized)
        img_idx += 1
    # OpenCV upscaled
    # plt.subplot(rows, 3, 3)
    # plt.imshow(resized[:, :, ::-1])
    plt.show()


def Test2():
    img_path = "/Users/hezhenbang/Desktop/UBCO/DL-COSC519/GroupProject/Code/jersey-2023/test/images/0/0_6.jpg"
    img_path_no_ext, img_ext = get_filename_without_extension(img_path)
    img = cv2.imread(img_path)
    img_size = img.shape

    for model_name, model in model_map.items():
        result = super_res_single_image(img, model_name, preserve_size=False)
        save_path = f'{img_path_no_ext}_{model_name}.{img_ext}'
        cv2.imwrite(save_path, result)

Test2()