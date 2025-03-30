import shutil

import cv2
import os
import numpy as np
import time
import json
import configuration
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from argparse import ArgumentParser
from xtcocotools.coco import COCO
import concurrent.futures

from configuration import openpose_bin_dir

#                     0      1        2       3       4       5       6       7       8       9       10     11      12      13       14      15      16     17
coco17_keypoints = ['Nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
coco18_keypoints2 = ['Nose', 'Neck', 'Lsho', 'Lelb', 'Lwri', 'Rsho', 'Relb', 'Rwri','Lhip', 'Lkne', 'Lank', 'Rhip', 'Rkne', 'Rank', 'Reye', 'Leye', 'Rear', 'Lear']
coco18_keypoints = ['Nose', 'Neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

coco18_to_coco17_idx_map = []
coco17_to_coco18_idx_map = []

def COCO18_to_COCO17(keypoints18):
    if len(keypoints18) != 18*3:
        print(f'COCO18_to_COCO17(): Wanring: length of keypoints18 is not 18*3: {len(keypoints18)}')
    if coco18_to_coco17_idx_map is None or len(coco18_to_coco17_idx_map) == 0:
        for i in range(0, len(coco17_keypoints)):
            pose_name = coco17_keypoints[i]
            for j in range(0, len(coco18_keypoints)):
                coco18_idx = -1
                if coco18_keypoints[j] == pose_name:
                    coco18_idx = j
                    print(f'map coco17 {i} ({pose_name}) --> coco18 {coco18_idx}({coco18_keypoints[j]})')
                    coco18_to_coco17_idx_map.append(coco18_idx)
        map_str = ''
        for i in range(0, len(coco18_to_coco17_idx_map)):
            map_str += f'{coco18_to_coco17_idx_map[i]},'
        print(f'coco18_to_coco17_idx_map = [{map_str[:-1]}]')
    keypoints18 = np.array(keypoints18).reshape(-1, 3)
    keypoints17 = keypoints18[coco18_to_coco17_idx_map]
    return keypoints17.tolist() #keypoints17.flatten().tolist()

def main():
    parser = ArgumentParser()
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--out-json',
        type=str,
        default='',
        help='Json file containing results.')
    parser.add_argument(
        '--crop-padding-horizontal',
        type=int,
        default=20,
        help='Horizontal padding for cropping.')

    parser.add_argument(
        '--thread-num',
        type=int,
        default=1,
        help='Thread pool size')

    parser.add_argument(
        '--crop-padding-vertical',
        type=int,
        default=5,
        help='Vertical padding for cropping.')

    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. ')
    parser.add_argument(
        '--data-part',
        type=str,
        help='Dataset type')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()
    coco = COCO(args.json_file)
    img_keys = list(coco.imgs.keys())
    start_time = time.time()

    cwd_backup = os.getcwd()
    print(f'current working directory: {cwd_backup}, now change to "openpose"')
    os.chdir('openpose')

    parent_dirs = set()
    for img_key in img_keys:
        image = coco.loadImgs(img_key)[0]
        image_path = os.path.join(args.img_root, image['file_name'])
        image_par_path = os.path.normpath(os.path.dirname(image_path))
        if image_par_path not in parent_dirs and os.path.exists(image_par_path) and os.path.isdir(image_par_path):
            parent_dirs.add(image_par_path)

    print(f'Extracted {len(parent_dirs)} folders')
    temp_dir = f'./openpose-cache/{args.data_part}'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f'mkdir {temp_dir}')
    import platform
    openpose_bin_dir = configuration.openpose_bin_dir
    if platform.system() == 'Windows':
        openpose_bin_dir = openpose_bin_dir.replace('/', '\\')
        print(f'Current platform is Windows, reformat openpose_bin_dir to {openpose_bin_dir}')
    print(f'Ready to perform pose estimation on {len(parent_dirs)} folders')
    if configuration.openpose_use_cache:
        print('OpenPose cache will be used, if you want to regenerate poses please delete cache folders for set configuration.openpose_use_cache to False')
    for _dir in tqdm(parent_dirs):
        basename = os.path.basename(_dir)
        out_dir = os.path.join(temp_dir, basename)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif configuration.openpose_use_cache:
            continue
        command = f'{openpose_bin_dir} --image_dir {_dir} --display 0 --render_pose 0 --model_pose COCO --write_json {out_dir}/' # --write_images
        os.system(command)

    ###################################
    not_processed_files = []
    unprocessed_temp_dir =  f'./openpose-cache/unprocessed_imgs/{args.data_part}'
    if os.path.exists(unprocessed_temp_dir):
        shutil.rmtree(unprocessed_temp_dir)
    os.makedirs(unprocessed_temp_dir)
    for image_id in tqdm(range(len(img_keys)), desc="Processing"):
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])

        image_parent_dir = os.path.dirname(image_name)
        image_parent_basename = os.path.basename(image_parent_dir)
        out_dir = os.path.join(temp_dir, image_parent_basename)

        image_basename = os.path.basename(image_name)
        filename_without_ext, extension = os.path.splitext(image_basename)
        openpose_json_path = os.path.join(unprocessed_temp_dir, filename_without_ext + '_keypoints.json')
        print(f'Read openpose json from {openpose_json_path} for {image_name}')
        if not os.path.exists(openpose_json_path):
            not_processed_files.append(image_name)
            shutil.copy(image_name, unprocessed_temp_dir)
    if len(not_processed_files) > 0:
        print(f'There are still unprocessed images: {len(not_processed_files)} images, begin to perform pose estimation')
        command = f'{openpose_bin_dir} --image_dir {not_processed_files} --display 0 --render_pose 0 --model_pose COCO --write_json {not_processed_files}/'  # --write_images
        # print(command)
        os.system(command)
        print(f'Pose estimation ended')
    ###################################
    results = []
    for image_id in tqdm(range(len(img_keys)), desc="Processing"):
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])

        image_parent_dir = os.path.dirname(image_name)
        image_parent_basename = os.path.basename(image_parent_dir)
        out_dir = os.path.join(temp_dir, image_parent_basename)

        image_basename = os.path.basename(image_name)
        filename_without_ext, extension = os.path.splitext(image_basename)
        openpose_json_path = os.path.join(args.out_dir, filename_without_ext + '_keypoints.json')
        print(f'Read openpose json from {openpose_json_path} for {image_name}')
        if not os.path.exists(openpose_json_path):
            old_json = openpose_json_path
            openpose_json_path = os.path.join(unprocessed_temp_dir, filename_without_ext + '_keypoints.json')
            print(f'Original openpose json: {old_json} does not exist, use newly created json {openpose_json_path}')
        with open(openpose_json_path, 'r') as f:
            json_data = json.load(f)
            people = json_data['people']
            keypoints18 = people[0]['pose_keypoints_2d']
            keypoints17 = COCO18_to_COCO17(keypoints18)
        result = {"img_name": image['file_name'], "id": image_id, "keypoints": keypoints17}
        results.append(result)
    pass


    if args.out_json != '':
        with open(args.out_json, 'w') as fp:
            print(f'openpose.py: main() writing results to {args.out_json}...')
            json.dump({"pose_results": results}, fp)
    else:
        print(f'openpose.py: main() there is no valid output path given!')
    end_time = time.time()
    print(f'openpose.py: main() took {end_time - start_time:.4f} seconds')
    os.chdir(cwd_backup)
    print(f'Change cwd back to {cwd_backup}')

if __name__ == "__main__":
    main()
    pass